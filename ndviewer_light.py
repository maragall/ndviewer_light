"""
Lightweight NDV-based viewer

Supports: OME-TIFF and single-TIFF acquisitions with lazy loading via dask.
Lazy loading enables fast initial display by only reading image planes on-demand.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING


import numpy as np


from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QMainWindow,
    QFileDialog,
    QApplication,
    QStyleFactory,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor

if TYPE_CHECKING:
    import xarray as xr

# Constants
TIFF_EXTENSIONS = {".tif", ".tiff"}
LIVE_REFRESH_INTERVAL_MS = 750

logger = logging.getLogger(__name__)

# Module-level variable for voxel scale (used by monkey-patched add_volume)
_current_voxel_scale: Optional[Tuple[float, float, float]] = None

# NDV viewer
try:
    import ndv

    NDV_AVAILABLE = True
    # Monkeypatch superqt slider to respect full ranges (self-contained)
    try:
        from superqt.sliders import QLabeledSlider

        if not getattr(QLabeledSlider, "_ndv_range_patch", False):
            _orig_setRange = QLabeledSlider.setRange

            def _patched_setRange(self, a, b):
                _orig_setRange(self, a, b)
                if hasattr(self, "_slider"):
                    self._slider.setMinimum(a)
                    self._slider.setMaximum(b)
                if hasattr(self, "_label"):
                    try:
                        self._label.setRange(a, b)
                    except Exception as e:
                        logger.debug("Failed to set label range: %s", e)

            QLabeledSlider.setRange = _patched_setRange
            QLabeledSlider._ndv_range_patch = True
    except ImportError:
        pass  # superqt not available

    # Monkeypatch vispy VolumeVisual to support anisotropic voxels
    # This allows correct 3D rendering when Z step differs from XY pixel size
    try:
        from ndv.views._vispy._array_canvas import VispyArrayCanvas
        from vispy.visuals.volume import VolumeVisual

        if not getattr(VolumeVisual, "_voxel_scale_patch", False):
            _orig_init = VolumeVisual.__init__
            _orig_create_vertex_data = VolumeVisual._create_vertex_data

            def _patched_init(self, *args, **kwargs):
                """Initialize VolumeVisual and capture the current voxel scale.

                Storing the scale as an instance attribute ensures thread safety
                when multiple volumes are created concurrently - each volume
                captures the scale that was active at its construction time.
                """
                _orig_init(self, *args, **kwargs)
                # Capture the voxel scale active at construction time
                global _current_voxel_scale
                self._voxel_scale = _current_voxel_scale

            VolumeVisual.__init__ = _patched_init

            def _patched_create_vertex_data(self):
                """Create vertices with Z scaling for anisotropic voxels.

                Uses the instance's _voxel_scale attribute (set at construction)
                rather than the global to ensure correct scaling even when
                multiple volumes exist with different scales.

                Falls back to original implementation when no scale is set.
                """
                # If no scale set, use original implementation
                scale = getattr(self, "_voxel_scale", None)
                if scale is None:
                    return _orig_create_vertex_data(self)

                shape = self._vol_shape

                # Get corner coordinates with Z scaling
                x0, x1 = -0.5, shape[2] - 0.5
                y0, y1 = -0.5, shape[1] - 0.5

                # Apply Z scale from instance attribute
                sz = scale[2]
                z0, z1 = -0.5 * sz, (shape[0] - 0.5) * sz

                pos = np.array(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x0, y1, z0],
                        [x1, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x0, y1, z1],
                        [x1, y1, z1],
                    ],
                    dtype=np.float32,
                )

                indices = np.array(
                    [2, 6, 0, 4, 5, 6, 7, 2, 3, 0, 1, 5, 3, 7], dtype=np.uint32
                )

                self._vertices.set_data(pos)
                self._index_buffer.set_data(indices)

            VolumeVisual._create_vertex_data = _patched_create_vertex_data
            VolumeVisual._voxel_scale_patch = True
            logger.info("Voxel scale patch applied to VolumeVisual")

        # Also patch add_volume to update camera range
        if not getattr(VispyArrayCanvas, "_camera_scale_patch", False):
            _orig_add_volume = VispyArrayCanvas.add_volume

            def _patched_add_volume(self, data=None):
                global _current_voxel_scale
                handle = _orig_add_volume(self, data)
                # Update camera to account for scaled Z dimension
                if _current_voxel_scale is not None and data is not None:
                    # Ensure data has at least 3 dimensions
                    shape = getattr(data, "shape", None)
                    if shape is None or len(shape) < 3:
                        return handle
                    try:
                        sz = _current_voxel_scale[2]
                        if abs(sz - 1.0) > 0.01:
                            z_size = shape[0] * sz
                            max_size = max(shape[1], shape[2], z_size)
                            # Add margin to scale_factor for comfortable viewing distance
                            self._camera.scale_factor = max_size + 6
                            self._view.camera.set_range(
                                x=(0, shape[2]),
                                y=(0, shape[1]),
                                z=(0, z_size),
                                margin=0.01,
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to adjust camera for anisotropic voxels: %s", e
                        )
                return handle

            VispyArrayCanvas.add_volume = _patched_add_volume
            VispyArrayCanvas._camera_scale_patch = True
    except ImportError:
        pass  # vispy not available

except ImportError:
    NDV_AVAILABLE = False

# Lazy loading
try:
    import tifffile as tf
    import xarray as xr
    import dask.array as da
    from functools import lru_cache
    from scipy.ndimage import zoom as ndimage_zoom

    LAZY_LOADING_AVAILABLE = True
except ImportError:
    LAZY_LOADING_AVAILABLE = False

# OpenGL 3D texture size limit (conservative estimate for most GPUs)
MAX_3D_TEXTURE_SIZE = 2048

# Channel label update retry configuration
CHANNEL_LABEL_UPDATE_MAX_RETRIES = 20
CHANNEL_LABEL_UPDATE_RETRY_DELAY_MS = 100

# Register custom DataWrapper for automatic 3D downsampling
if NDV_AVAILABLE and LAZY_LOADING_AVAILABLE:
    from ndv.models._data_wrapper import XarrayWrapper
    from collections.abc import Mapping

    class Downsampling3DXarrayWrapper(XarrayWrapper):
        """XarrayWrapper that automatically downsamples 3D volumes for OpenGL.

        This wrapper extends NDV's XarrayWrapper to detect when a 3D volume
        request would exceed OpenGL texture limits and automatically downsamples
        the data. 2D slice requests remain at full resolution.
        """

        # Higher priority than default XarrayWrapper (50)
        PRIORITY = 40

        # Class-level cache for OpenGL texture limit (queried once, shared by all instances)
        _cached_max_texture_size: Optional[int] = None

        def __init__(self, data: xr.DataArray):
            super().__init__(data)

        @classmethod
        def _get_max_texture_size(cls) -> int:
            """Query and cache the GPU's GL_MAX_3D_TEXTURE_SIZE.

            This is queried lazily on first 3D request when OpenGL context exists.
            Falls back to conservative default if query fails or no context available.
            """
            if cls._cached_max_texture_size is None:
                try:
                    # Check if vispy has an active GL context before querying
                    # Calling OpenGL without a context causes segfault
                    from vispy import app

                    # Check for active vispy application - use _backend_module which
                    # is set when a backend is actually loaded and initialized
                    backend = getattr(app, "_backend_module", None)
                    if backend is None:
                        logger.debug(
                            "No vispy backend loaded - using fallback texture size"
                        )
                        cls._cached_max_texture_size = MAX_3D_TEXTURE_SIZE
                        return cls._cached_max_texture_size

                    from OpenGL.GL import glGetIntegerv, GL_MAX_3D_TEXTURE_SIZE

                    limit = glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)
                    cls._cached_max_texture_size = int(limit)
                    logger.info(f"Detected GL_MAX_3D_TEXTURE_SIZE: {limit}")
                except Exception as e:
                    logger.debug(f"Failed to query GL_MAX_3D_TEXTURE_SIZE: {e}")
                    cls._cached_max_texture_size = MAX_3D_TEXTURE_SIZE  # Fallback
            return cls._cached_max_texture_size

        @classmethod
        def supports(cls, obj) -> bool:
            """Check if this wrapper supports the given object."""
            # Note: LAZY_LOADING_AVAILABLE check is unnecessary here since this
            # class is only defined when LAZY_LOADING_AVAILABLE is True (line 85)
            return isinstance(obj, xr.DataArray)

        def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
            """Return a slice of the data, with automatic 3D downsampling.

            For 2D slices (viewing a single plane), returns full resolution.
            For 3D volumes (viewing a stack), downsamples if needed to fit
            within OpenGL texture limits.

            Downsampling strategy:
            - If physical pixel sizes are known (pixel_size_um, dz_um in attrs),
              scale to maintain correct physical aspect ratio
            - Otherwise: z scaled independently, x/y scaled uniformly
            - channel/time/fov: never scaled
            """
            # Get the data using parent's implementation
            data = super().isel(index)

            # Determine which original dimensions are non-singleton
            dims = self._data.dims
            non_singleton_dims = []
            for i, dim in enumerate(dims):
                idx = index.get(i, slice(None))
                if isinstance(idx, slice):
                    dim_size = self._data.shape[i]
                    start = idx.start or 0
                    stop = idx.stop or dim_size
                    if stop - start > 1:
                        non_singleton_dims.append(str(dim).lower())

            # Check if we have spatial z dimension (indicates 3D volume)
            spatial_z_names = {"z", "z_level", "depth", "focus"}
            has_z = any(d in spatial_z_names for d in non_singleton_dims)
            if not has_z:
                return data  # Not a 3D volume request

            # Check if any spatial dimension exceeds the texture limit
            max_texture_size = self._get_max_texture_size()

            # First pass: find dimensions and their sizes in output data
            dim_info = []  # [(dim_name, size), ...]
            for i, dim in enumerate(dims):
                idx = index.get(i, slice(None))
                if isinstance(idx, int):
                    continue  # Dropped dimension
                dim_info.append((str(dim).lower(), data.shape[len(dim_info)]))

            # Compute zoom factors for downsampling
            zoom_factors, needs_downsampling = self._compute_simple_zoom_factors(
                dim_info, max_texture_size, spatial_z_names
            )

            if needs_downsampling:
                logger.info(
                    f"Downsampling 3D volume from {data.shape} "
                    f"(factors={[f'{z:.3f}' for z in zoom_factors]}) for OpenGL rendering"
                )

                # Use order=0 (nearest neighbor) for speed - much faster than bilinear
                try:
                    downsampled = ndimage_zoom(data, zoom_factors, order=0)
                    return downsampled.astype(data.dtype)
                except Exception as e:
                    logger.warning(f"Downsampling failed: {e}, returning original data")
                    return data

            return data

        def _compute_simple_zoom_factors(
            self, dim_info: list, max_texture_size: int, spatial_z_names: set
        ) -> tuple:
            """Compute zoom factors for 3D volume downsampling.

            Strategy:
            - XY: scaled uniformly (same factor for X and Y) to preserve XY aspect ratio
            - Z: scaled independently only if it exceeds the texture limit
            - Non-spatial dims (channel, time, fov): never scaled

            Note: Physical aspect ratio correction is handled via vertex scaling
            in the vispy VolumeVisual patch.
            """
            # Calculate xy scale factor (uniform for x and y to preserve XY aspect)
            xy_sizes = [size for name, size in dim_info if name in {"y", "x"}]
            xy_max = max(xy_sizes) if xy_sizes else 0
            xy_scale = max_texture_size / xy_max if xy_max > max_texture_size else 1.0

            # Build zoom factors
            zoom_factors = []
            needs_downsampling = False
            for dim_name, dim_size in dim_info:
                if dim_name in {"y", "x"}:
                    zoom_factors.append(xy_scale)
                    if xy_scale < 1.0:
                        needs_downsampling = True
                elif dim_name in spatial_z_names and dim_size > max_texture_size:
                    z_scale = max_texture_size / dim_size
                    zoom_factors.append(z_scale)
                    needs_downsampling = True
                else:
                    zoom_factors.append(1.0)

            return zoom_factors, needs_downsampling


# Filename patterns (from common.py)
FPATTERN = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE
)
FPATTERN_OME = re.compile(r"(?P<r>[^_]+)_(?P<f>\d+)\.ome\.tiff?", re.IGNORECASE)


# Helper functions
def extract_wavelength(channel_str: str):
    """Extract wavelength (nm) from channel string; None if unknown."""
    if not channel_str:
        return None
    lower = channel_str.lower()
    if re.fullmatch(r"ch\d+", lower):
        return None
    # Direct wavelength pattern
    if m := re.search(r"(\d{3,4})[ _]*nm", channel_str, re.IGNORECASE):
        return int(m.group(1))

    # Common fluorophores
    fluor_map = {
        "dapi": 405,
        "hoechst": 405,
        "gfp": 488,
        "fitc": 488,
        "alexa488": 488,
        "tritc": 561,
        "cy3": 561,
        "mcherry": 561,
        "cy5": 640,
        "alexa647": 640,
        "cy7": 730,
    }
    channel_lower = channel_str.lower()
    for fluor, wl in fluor_map.items():
        if fluor in channel_lower:
            return wl

    # Fallback
    numbers = re.findall(r"\d{3,4}", channel_str)
    if numbers:
        # Prefer the last 3-4 digit group (likely wavelength)
        val = int(numbers[-1])
        return val if val > 0 else None
    return None


def extract_ome_physical_sizes(
    ome_metadata: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract physical pixel sizes from OME-XML metadata.

    Returns:
        Tuple of (pixel_size_x_um, pixel_size_y_um, pixel_size_z_um).
        Values are in micrometers. None if not found or unable to parse.
    """
    if not ome_metadata:
        return None, None, None

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(ome_metadata)
        # Try multiple OME namespace versions
        namespaces = [
            {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"},
            {"ome": "http://www.openmicroscopy.org/Schemas/OME/2015-01"},
            {"ome": "http://www.openmicroscopy.org/Schemas/OME/2013-06"},
            {},  # No namespace fallback
        ]

        for ns in namespaces:
            # Find Pixels element which contains physical size attributes
            if ns:
                pixels = root.find(".//ome:Pixels", ns)
            else:
                # Try without or with any namespace (.//{*} matches any namespace)
                pixels = root.find(".//{*}Pixels")

            if pixels is not None:
                size_x = pixels.get("PhysicalSizeX")
                size_y = pixels.get("PhysicalSizeY")
                size_z = pixels.get("PhysicalSizeZ")
                unit_x = pixels.get("PhysicalSizeXUnit", "µm")
                unit_y = pixels.get("PhysicalSizeYUnit", "µm")
                unit_z = pixels.get("PhysicalSizeZUnit", "µm")

                def to_micrometers(value: Optional[str], unit: str) -> Optional[float]:
                    if value is None:
                        return None
                    try:
                        val = float(value)
                        # Convert to micrometers based on unit
                        unit_lower = unit.lower()
                        if unit_lower in ("nm", "nanometer", "nanometers"):
                            result = val / 1000.0
                        elif unit_lower in ("mm", "millimeter", "millimeters"):
                            result = val * 1000.0
                        elif unit_lower in ("m", "meter", "meters"):
                            result = val * 1e6
                        else:
                            # Default assumes micrometers (µm, um, micron, etc.)
                            result = val
                        # Physical sizes must be strictly positive
                        if result <= 0:
                            return None
                        return result
                    except (ValueError, TypeError):
                        return None

                px = to_micrometers(size_x, unit_x)
                py = to_micrometers(size_y, unit_y)
                pz = to_micrometers(size_z, unit_z)

                if px is not None or py is not None or pz is not None:
                    return px, py, pz

    except Exception as e:
        logger.debug("Failed to extract physical sizes from OME metadata: %s", e)

    return None, None, None


def read_acquisition_parameters(
    base_path: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Read pixel size and dz from acquisition parameters JSON file.

    Supports both "acquisition_parameters.json" and "acquisition parameters.json".
    Can compute pixel size from sensor_pixel_size_um and objective magnification.

    Returns:
        Tuple of (pixel_size_um, dz_um). None if not found.
    """
    # Try both filename variants
    params_file = base_path / "acquisition_parameters.json"
    if not params_file.exists():
        params_file = base_path / "acquisition parameters.json"
    if not params_file.exists():
        return None, None

    try:
        with open(params_file, "r") as f:
            params = json.load(f)

        pixel_size = None
        dz = None

        # Try direct pixel size keys first
        for key in ["pixel_size_um", "pixel_size", "pixelSize", "pixel_size_xy"]:
            if key in params:
                try:
                    candidate_pixel = float(params[key])
                except (TypeError, ValueError):
                    continue
                # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                # Range 0.01-100 µm covers most use cases including low-mag imaging
                if 0.01 < candidate_pixel <= 100:
                    pixel_size = candidate_pixel
                    break

        # If not found, compute from sensor pixel size and magnification
        # Account for tube lens ratio: actual_mag = nominal_mag × (tube_lens / obj_tube_lens)
        if pixel_size is None:
            sensor_pixel = params.get("sensor_pixel_size_um")
            objective = params.get("objective", {})
            if isinstance(objective, dict):
                nominal_mag = objective.get("magnification")
                obj_tube_lens = objective.get("tube_lens_f_mm")
            else:
                nominal_mag = None
                obj_tube_lens = None
            tube_lens = params.get("tube_lens_mm")

            if sensor_pixel is not None and nominal_mag is not None and nominal_mag > 0:
                # Compute actual magnification with tube lens correction
                if (
                    tube_lens is not None
                    and obj_tube_lens is not None
                    and obj_tube_lens > 0
                ):
                    actual_mag = float(nominal_mag) * (
                        float(tube_lens) / float(obj_tube_lens)
                    )
                else:
                    actual_mag = float(nominal_mag)
                computed = float(sensor_pixel) / actual_mag
                # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                # Range 0.01-100 µm covers most use cases including low-mag imaging
                if 0.01 < computed <= 100:
                    pixel_size = computed

        # Try common key names for z spacing
        for key in [
            "dz_um",
            "dz",
            "z_step",
            "zStep",
            "z_spacing",
            "pixel_size_z",
            "dz(um)",
        ]:
            if key in params:
                try:
                    candidate_dz = float(params[key])
                except (TypeError, ValueError):
                    continue
                # dz must be strictly positive to be physically meaningful
                if candidate_dz > 0:
                    dz = candidate_dz
                    break

        return pixel_size, dz

    except Exception as e:
        logger.debug("Failed to read acquisition parameters: %s", e)
        return None, None


def read_tiff_pixel_size(tiff_path: str) -> Optional[float]:
    """Read pixel size from TIFF metadata tags.

    Attempts to extract pixel size from (in priority order):
    1. ImageDescription tag (JSON metadata from some microscopy software)
    2. XResolution/YResolution tags with ResolutionUnit

    Note on ResolutionUnit: Only inch (2) and centimeter (3) units are supported.
    Unit value 1 ("no absolute unit") is explicitly rejected because it cannot
    be reliably converted to physical units. Many image editors set resolution
    tags without meaningful physical units, so we require explicit inch/cm units.

    Returns:
        Pixel size in micrometers, or None if not found.
    """
    if not LAZY_LOADING_AVAILABLE:
        return None

    try:
        with tf.TiffFile(tiff_path) as tif:
            page = tif.pages[0]

            # Try ImageDescription tag FIRST for JSON metadata
            # (more reliable for microscopy data)
            desc = page.tags.get("ImageDescription")
            if desc is not None:
                desc_str = desc.value
                if isinstance(desc_str, bytes):
                    desc_str = desc_str.decode("utf-8", errors="ignore")

                # Try to parse as JSON
                try:
                    metadata = json.loads(desc_str)
                    for key in [
                        "pixel_size_um",
                        "pixel_size",
                        "PixelSize",
                        "pixelSize",
                    ]:
                        if key in metadata:
                            val = float(metadata[key])
                            # Require strictly positive value
                            if val <= 0:
                                continue
                            # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                            # Range 0.01-100 µm covers most use cases including low-mag imaging
                            if 0.01 < val <= 100:
                                return val
                except (json.JSONDecodeError, ValueError, TypeError):
                    # JSON parsing failed; fall through to resolution tags below
                    pass

            # Try XResolution/YResolution tags with proper unit
            x_res = page.tags.get("XResolution")
            res_unit = page.tags.get("ResolutionUnit")

            # Only use resolution tags if we have a proper unit (inch=2 or cm=3)
            unit_value = res_unit.value if res_unit else 1
            if unit_value not in (2, 3):
                return None  # No unit or unknown unit - can't reliably convert

            if x_res is not None:
                # XResolution is stored as a fraction (numerator, denominator)
                x_res_value = x_res.value
                if isinstance(x_res_value, tuple) and len(x_res_value) == 2:
                    pixels_per_unit = x_res_value[0] / x_res_value[1]
                else:
                    pixels_per_unit = float(x_res_value)

                # Skip default/invalid values (must be > 1 to be meaningful)
                if pixels_per_unit <= 1:
                    return None

                # Convert to micrometers based on unit
                if unit_value == 2:  # inch
                    # pixels/inch -> um/pixel: 25400 um/inch / pixels_per_inch
                    pixel_size_um = 25400.0 / pixels_per_unit
                else:  # centimeter (unit_value == 3)
                    # pixels/cm -> um/pixel: 10000 um/cm / pixels_per_cm
                    pixel_size_um = 10000.0 / pixels_per_unit

                # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                # Range 0.01-100 µm covers most use cases including low-mag imaging
                if 0.01 < pixel_size_um <= 100:
                    return pixel_size_um

    except Exception as e:
        logger.debug("Failed to read pixel size from TIFF tags: %s", e)

    return None


def detect_format(base_path: Path) -> str:
    """Detect OME-TIFF vs single-TIFF format."""
    ome_dir = base_path / "ome_tiff"
    if ome_dir.exists():
        if any(".ome" in f.name for f in ome_dir.glob("*.tif*")):
            return "ome_tiff"

    first_tp = next(
        (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()), None
    )
    if first_tp:
        if any(".ome" in f.name for f in first_tp.glob("*.tif*")):
            return "ome_tiff"
    return "single_tiff"


def wavelength_to_colormap(wavelength: Optional[int]) -> str:
    """Map wavelength to NDV colormap."""
    if wavelength is None or wavelength == 0:
        return "gray"
    if wavelength <= 420:
        return "blue"
    elif 470 <= wavelength <= 510:
        return "green"
    elif 540 <= wavelength <= 590:
        return "yellow"
    elif 620 <= wavelength <= 660:
        return "red"
    elif wavelength >= 700:
        return "magenta"
    return "gray"


def _apply_dark_theme(widget: QWidget) -> None:
    """Apply dark Fusion theme to a widget."""
    widget.setStyle(QStyleFactory.create("Fusion"))

    p = widget.palette()
    p.setColor(QPalette.Window, QColor(53, 53, 53))
    p.setColor(QPalette.WindowText, QColor(255, 255, 255))
    p.setColor(QPalette.Base, QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.Text, QColor(255, 255, 255))
    p.setColor(QPalette.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    widget.setPalette(p)


class LauncherWindow(QMainWindow):
    """Separate launcher window with dropbox for dataset selection."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDViewer Lightweight - Open Dataset")
        self.setGeometry(100, 100, 400, 300)  # 4:3 aspect, narrower
        self._set_dark_theme()

        central = QWidget()
        layout = QVBoxLayout()

        # Drop zone / Open button
        self.drop_label = QLabel("Drop folder here\nor click to open")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                padding: 40px;
                background: #2a2a2a;
                color: #aaa;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: #888;
                background: #333;
            }
        """
        )
        self.drop_label.setMinimumHeight(150)
        self.drop_label.mousePressEvent = lambda e: self._open_folder_dialog()
        layout.addWidget(self.drop_label)

        # Status
        self.status_label = QLabel("No dataset loaded")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        # layout.addWidget(self.status_label) # hide status label

        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setAcceptDrops(True)

        self.viewer_window = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._launch_viewer(path)

    def _open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self._launch_viewer(path)

    def _launch_viewer(self, path: str):
        """Launch main viewer window with dataset."""
        self.status_label.setText(f"Opening: {Path(path).name}...")
        QApplication.processEvents()

        # Keep launcher open; allow multiple drops without restarting.
        if self.viewer_window:
            try:
                self.viewer_window.close()
            except Exception as e:
                logger.debug("Failed to close previous viewer: %s", e)
        self.viewer_window = LightweightMainWindow(path)
        self.viewer_window.show()

    def _set_dark_theme(self):
        _apply_dark_theme(self)


class LightweightViewer(QWidget):
    """Minimal NDV-based viewer."""

    dataset_path: str
    ndv_viewer: Optional["ndv.ArrayViewer"]
    _xarray_data: Optional["xr.DataArray"]
    _open_handles: List
    _last_sig: Optional[tuple]
    _refresh_timer: Optional[QTimer]

    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.ndv_viewer = None
        self._xarray_data = None  # Store for external access
        self._open_handles = []  # Keep tif handles alive when mmap is used
        self._last_sig = None
        self._refresh_timer = None
        self._channel_label_generation = 0  # Generation counter for retry cancellation
        self._pending_channel_label_retries = (
            0  # Retry counter for channel label updates
        )
        self._setup_ui()
        self.load_dataset(dataset_path)
        self._setup_live_refresh()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Loading dataset...")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        # layout.addWidget(self.status_label)

        # NDV placeholder
        if NDV_AVAILABLE:
            dummy = np.zeros((1, 100, 100), dtype=np.uint16)
            self.ndv_viewer = ndv.ArrayViewer(
                dummy,
                channel_axis=0,
                channel_mode="composite",
                visible_axes=(-2, -1),
            )
            layout.addWidget(self.ndv_viewer.widget(), 1)
        else:
            placeholder = QLabel("NDV not available.\npip install ndv[vispy,pyqt]")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder, 1)

        self.setLayout(layout)

    def _setup_live_refresh(self):
        """Poll the dataset folder periodically to pick up new timepoints during acquisition."""
        # Only enable when lazy loading + NDV are available; otherwise refresh does nothing useful.
        if not (LAZY_LOADING_AVAILABLE and NDV_AVAILABLE and self.ndv_viewer):
            return
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(LIVE_REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._maybe_refresh)
        self._refresh_timer.start()

    def _close_open_handles(self):
        """Close mmap TiffFile handles (OME path) from the previously loaded dataset."""
        for h in getattr(self, "_open_handles", []) or []:
            try:
                h.close()
            except Exception as e:
                logger.debug("Failed to close TiffFile handle: %s", e)
        self._open_handles = []

    def closeEvent(self, event):
        """Clean up resources when the widget is closed."""
        if self._refresh_timer:
            self._refresh_timer.stop()
        self._close_open_handles()
        super().closeEvent(event)

    def _force_refresh(self):
        self._last_sig = None
        self._maybe_refresh()

    def _dataset_signature(self) -> tuple:
        """Return a cheap signature that changes when new data likely arrived."""
        base = Path(self.dataset_path)
        fmt = detect_format(base)

        if fmt == "single_tiff":
            tp_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
            if not tp_dirs:
                return (fmt, -1, 0, 0)

            t_vals = sorted(int(d.name) for d in tp_dirs)
            first_tp = base / str(t_vals[0])
            latest_tp = base / str(t_vals[-1])

            # FOVs are assumed to only appear in the first timepoint during acquisition.
            fov_set = set()
            try:
                if first_tp.exists():
                    for f in first_tp.iterdir():
                        if f.suffix.lower() not in TIFF_EXTENSIONS:
                            continue
                        m = FPATTERN.search(f.name)
                        if m:
                            fov_set.add((m.group("r"), int(m.group("f"))))
            except Exception as e:
                logger.debug("Error scanning FOVs: %s", e)

            # Count files in latest timepoint to detect when files are actually written
            # (not just when the folder is created)
            latest_file_count = 0
            try:
                if latest_tp.exists():
                    latest_file_count = sum(
                        1
                        for f in latest_tp.iterdir()
                        if f.suffix.lower() in TIFF_EXTENSIONS
                    )
            except Exception as e:
                logger.debug("Error counting files in latest timepoint: %s", e)

            return (fmt, max(t_vals), len(fov_set), latest_file_count)

        # ome_tiff
        ome_dir = base / "ome_tiff"
        if not ome_dir.exists():
            ome_dir = next(
                (d for d in base.iterdir() if d.is_dir() and d.name.isdigit()), base
            )

        ome_files = sorted(ome_dir.glob("*.ome.tif*"))
        n_ome = len(ome_files)
        t_len = -1
        st = None
        if ome_files:
            try:
                st = ome_files[0].stat()
            except Exception as e:
                logger.debug("Failed to stat OME file: %s", e)
                st = None
            try:
                with tf.TiffFile(str(ome_files[0])) as tif:
                    series = tif.series[0]
                    axes = series.axes
                    shape = series.shape
                    if "T" in axes:
                        t_len = int(shape[axes.index("T")])
                    else:
                        t_len = 1
            except Exception as e:
                # File may be mid-write; fall back on size/mtime if available
                logger.debug("Failed to read OME series (may be mid-write): %s", e)

        if st is None:
            return (fmt, n_ome, t_len)
        return (
            fmt,
            n_ome,
            t_len,
            st.st_size,
            getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
        )

    def _try_inplace_ndv_update(self, data: "xr.DataArray") -> bool:
        """Best-effort no-flicker data swap for ndv, depending on installed ndv version."""
        v = self.ndv_viewer
        if v is None:
            return False

        # Try common APIs across versions.
        candidates = [
            ("set_data", True),
            ("setData", True),
            ("set_array", True),
            ("setArray", True),
        ]
        for name, is_call in candidates:
            attr = getattr(v, name, None)
            if callable(attr):
                try:
                    attr(data)
                    return True
                except Exception:
                    pass

        # Try common attribute assignment patterns.
        for prop in ["data", "array"]:
            if hasattr(v, prop):
                try:
                    setattr(v, prop, data)
                    return True
                except Exception:
                    pass

        # Some viewers tuck the model under .viewer or .model
        for inner_name in ["viewer", "model"]:
            inner = getattr(v, inner_name, None)
            if inner is None:
                continue
            for name in ["set_data", "setData", "set_array", "setArray"]:
                fn = getattr(inner, name, None)
                if callable(fn):
                    try:
                        fn(data)
                        return True
                    except Exception:
                        pass
            for prop in ["data", "array"]:
                if hasattr(inner, prop):
                    try:
                        setattr(inner, prop, data)
                        return True
                    except Exception:
                        pass

        return False

    def _maybe_refresh(self):
        if not LAZY_LOADING_AVAILABLE:
            return

        try:
            sig = self._dataset_signature()
        except Exception as e:
            logger.debug("Failed to compute dataset signature: %s", e)
            return
        if sig == self._last_sig:
            return
        self._last_sig = sig

        data = self._create_lazy_array(Path(self.dataset_path))
        if data is None:
            return

        # Swap dataset, keeping OME handles alive for the new data
        old_handles = getattr(self, "_open_handles", [])
        old_data = self._xarray_data
        self._xarray_data = data
        self._open_handles = data.attrs.get("_open_tifs", [])

        # Check if data structure changed (channels, dims) - if so, force full rebuild
        structure_changed = self._data_structure_changed(old_data, data)

        # Prefer in-place update to avoid visible refresh, but only if structure unchanged.
        # When structure changes (e.g., different channels), we must rebuild the viewer
        # to avoid stale channel controls persisting from the previous dataset.
        if not structure_changed and self._try_inplace_ndv_update(data):
            # Update channel labels for the new data
            self._initiate_channel_label_update()
            # Close old handles after successful swap.
            for h in old_handles or []:
                try:
                    h.close()
                except Exception as e:
                    logger.debug("Failed to close old handle: %s", e)
            return

        # Fallback: rebuild widget (may be visible on some platforms). Reduce flicker a bit.
        try:
            self.setUpdatesEnabled(False)
            self._set_ndv_data(data)
        finally:
            self.setUpdatesEnabled(True)
            # Close old handles regardless.
            for h in old_handles or []:
                try:
                    h.close()
                except Exception as e:
                    logger.debug("Failed to close old handle: %s", e)

    def _data_structure_changed(
        self, old_data: Optional["xr.DataArray"], new_data: "xr.DataArray"
    ) -> bool:
        """Check if data structure changed significantly (requiring full viewer rebuild).

        This detects changes in dimensions, channel count, channel names, or LUT
        configuration that would require rebuilding the NDV viewer rather than
        just swapping data in-place.

        Returns:
            True if structure changed and viewer needs full rebuild.
        """
        if old_data is None:
            return True

        try:
            # Check if dims changed
            if old_data.dims != new_data.dims:
                return True

            # Check if channel count changed
            if "channel" in old_data.dims and "channel" in new_data.dims:
                old_channels = old_data.coords.get("channel", [])
                new_channels = new_data.coords.get("channel", [])
                if len(old_channels) != len(new_channels):
                    return True

            # Check if channel names changed
            old_names = old_data.attrs.get("channel_names", [])
            new_names = new_data.attrs.get("channel_names", [])
            if old_names != new_names:
                return True

            # Check if LUTs changed
            old_luts = old_data.attrs.get("luts", {})
            new_luts = new_data.attrs.get("luts", {})
            if old_luts != new_luts:
                return True

            return False
        except Exception as e:
            # On any error, assume structure changed to be safe
            logger.debug("Error checking data structure change: %s", e)
            return True

    def load_dataset(self, path: str):
        """Load dataset and display in NDV."""
        # Reset state when loading a new dataset to ensure clean slate.
        # This prevents stale channel controls from persisting when switching
        # between datasets with different channel configurations.
        self._last_sig = None
        self._xarray_data = None

        self.dataset_path = path
        self.status_label.setText(f"Loading: {Path(path).name}...")
        QApplication.processEvents()

        try:
            data = self._create_lazy_array(Path(path))
            if data is not None:
                self._xarray_data = data  # Store for profiling
                self._open_handles = data.attrs.get("_open_tifs", [])
                # Always do full rebuild when explicitly loading a new dataset.
                # This ensures channels/LUTs are properly reset.
                self._set_ndv_data(data)

                # Update status (keep it stable during live acquisition; avoid printing dims like time=...)
                self.status_label.setText(f"Loaded: {Path(path).name}")
            else:
                self.status_label.setText("Failed to load dataset")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            import traceback

            traceback.print_exc()

    def _create_lazy_array(self, base_path: Path) -> Optional[xr.DataArray]:
        """Create lazy xarray from dataset - auto-detects format."""
        if not LAZY_LOADING_AVAILABLE:
            return None

        fmt = detect_format(base_path)
        fovs = self._discover_fovs(base_path, fmt)

        if not fovs:
            print("No FOVs found")
            return None

        # print(f"Format: {fmt}, FOVs: {len(fovs)}")  # Disabled for profiling

        if fmt == "ome_tiff":
            return self._load_ome_tiff(base_path, fovs)
        else:
            return self._load_single_tiff(base_path, fovs)

    def _discover_fovs(self, base_path: Path, fmt: str) -> List[Dict]:
        """Discover all FOVs (region, fov) pairs."""
        fov_set = set()

        if fmt == "ome_tiff":
            ome_dir = base_path / "ome_tiff"
            if not ome_dir.exists():
                ome_dir = next(
                    (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                    base_path,
                )
            for f in ome_dir.glob("*.ome.tif*"):
                if m := FPATTERN_OME.search(f.name):
                    fov_set.add((m.group("r"), int(m.group("f"))))
        else:
            first_tp = next(
                (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                None,
            )
            if first_tp:
                for f in first_tp.glob("*.tiff"):
                    if m := FPATTERN.search(f.name):
                        fov_set.add((m.group("r"), int(m.group("f"))))

        return [{"region": r, "fov": f} for r, f in sorted(fov_set)]

    def _load_ome_tiff(
        self, base_path: Path, fovs: List[Dict]
    ) -> Optional[xr.DataArray]:
        """Fast OME-TIFF: mmap via tifffile.aszarr, small chunks, no big graphs."""
        try:
            ome_dir = base_path / "ome_tiff"
            if not ome_dir.exists():
                ome_dir = next(
                    (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                    base_path,
                )

            file_index = {}
            for f in ome_dir.glob("*.ome.tif*"):
                if m := FPATTERN_OME.search(f.name):
                    file_index[(m.group("r"), int(m.group("f")))] = str(f)
            if not file_index:
                return None

            first_file = next(iter(file_index.values()))
            with tf.TiffFile(first_file) as tif:
                series = tif.series[0]
                axes = series.axes
                shape = series.shape
                shape_dict = dict(zip(axes, shape))
                n_t = shape_dict.get("T", 1)
                n_c = shape_dict.get("C", 1)
                n_z = shape_dict.get("Z", 1)
                height = shape_dict.get("Y", shape[-2])
                width = shape_dict.get("X", shape[-1])
                channel_names = []
                pixel_size_x, pixel_size_y, pixel_size_z = None, None, None
                try:
                    if tif.ome_metadata:
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(tif.ome_metadata)
                        ns = {
                            "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                        }
                        for ch in root.findall(".//ome:Channel", ns):
                            name = ch.get("Name") or ch.get("ID", "")
                            if name:
                                channel_names.append(name)

                        # Extract physical pixel sizes
                        pixel_size_x, pixel_size_y, pixel_size_z = (
                            extract_ome_physical_sizes(tif.ome_metadata)
                        )
                except Exception as e:
                    logger.debug("Failed to parse OME metadata: %s", e)

            axis_map = {"T": "time", "Z": "z", "C": "channel", "Y": "y", "X": "x"}
            dims_base = [axis_map.get(ax, f"ax_{ax}") for ax in axes]
            # Build channel name list - prefer extracted names, fill/truncate to match n_c
            if not channel_names:
                channel_names = [f"Ch{i}" for i in range(n_c)]
            elif len(channel_names) < n_c:
                channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
            elif len(channel_names) > n_c:
                channel_names = channel_names[:n_c]
            # Keep coordinates numeric (indices) for all axes, including "channel";
            # channel names are stored in attrs and applied via _lut_controllers.
            # This convention is used consistently for both OME-TIFF and single-TIFF paths.
            coords_base = {
                axis_map.get(ax, f"ax_{ax}"): list(range(dim))
                for ax, dim in zip(axes, shape)
            }

            # Per-axis chunking: 1 for non-spatial, full for spatial
            chunks = []
            for ax, dim in zip(axes, shape):
                if ax in ("X", "Y"):
                    chunks.append(dim)
                else:
                    chunks.append(1)

            luts = {
                i: wavelength_to_colormap(extract_wavelength(name))
                for i, name in enumerate(channel_names)
            }
            n_fov = len(fovs)

            def open_zarr(path: str):
                tif = tf.TiffFile(path)
                zarr_store = tif.series[0].aszarr()
                return tif, zarr_store

            # One dask array per FOV, chunked per plane for fast single-slice reads
            fov_arrays = []
            tifs_kept = []
            for fov_idx in range(n_fov):
                region, fov = fovs[fov_idx]["region"], fovs[fov_idx]["fov"]
                filepath = file_index.get((region, fov))
                if not filepath:
                    fov_arrays.append(
                        da.zeros((n_t, n_z, n_c, height, width), dtype=np.uint16)
                    )
                    continue
                tif, zarr_store = open_zarr(filepath)
                tifs_kept.append(tif)
                arr = da.from_zarr(zarr_store, chunks=tuple(chunks))
                # keep tif open to support mmap; rely on Python GC after viewer closes
                fov_arrays.append(arr)

            # Insert fov axis immediately after time if present, else at front
            if "time" in dims_base:
                fov_axis = dims_base.index("time") + 1
            else:
                fov_axis = 0
            full_array = da.stack(fov_arrays, axis=fov_axis)

            dims_full = dims_base[:fov_axis] + ["fov"] + dims_base[fov_axis:]
            coords_full = coords_base.copy()
            coords_full["fov"] = list(range(n_fov))

            xarr = xr.DataArray(full_array, dims=dims_full, coords=coords_full)
            # Ensure standard dims exist with singleton axes if missing
            for ax in ["time", "fov", "z", "channel", "y", "x"]:
                if ax not in xarr.dims:
                    xarr = xarr.expand_dims({ax: [0]})
            xarr = xarr.transpose("time", "fov", "z", "channel", "y", "x")
            xarr.attrs["luts"] = luts
            xarr.attrs["channel_names"] = channel_names
            xarr.attrs["_open_tifs"] = tifs_kept

            # Store physical pixel sizes (in micrometers)
            if pixel_size_x is not None:
                xarr.attrs["pixel_size_x_um"] = pixel_size_x
            if pixel_size_y is not None:
                xarr.attrs["pixel_size_y_um"] = pixel_size_y
            if pixel_size_z is not None:
                xarr.attrs["pixel_size_z_um"] = pixel_size_z
            # Also store commonly used aliases
            if pixel_size_x is not None and pixel_size_y is not None:
                # Use average for isotropic XY pixel size
                xarr.attrs["pixel_size_um"] = (pixel_size_x + pixel_size_y) / 2
            if pixel_size_z is not None:
                xarr.attrs["dz_um"] = pixel_size_z

            return xarr
        except Exception as e:
            print(f"OME-TIFF load error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _load_single_tiff(
        self, base_path: Path, fovs: List[Dict]
    ) -> Optional[xr.DataArray]:
        """Fast single-TIFF: per-plane on-demand loads with tiny LRU cache."""
        try:
            file_index = {}  # (t, region, fov, z, channel) -> filepath
            t_set, z_set, c_set = set(), set(), set()

            for tp_dir in sorted(base_path.iterdir()):
                if not (tp_dir.is_dir() and tp_dir.name.isdigit()):
                    continue
                t = int(tp_dir.name)
                # Only add timepoint to t_set if it has at least one valid file
                # This prevents showing black images for empty/incomplete timepoints
                has_files = False
                for f in tp_dir.iterdir():
                    if f.suffix.lower() not in TIFF_EXTENSIONS:
                        continue
                    if m := FPATTERN.search(f.name):
                        region, fov = m.group("r"), int(m.group("f"))
                        z, channel = int(m.group("z")), m.group("c")
                        z_set.add(z)
                        c_set.add(channel)
                        file_index[(t, region, fov, z, channel)] = str(f)
                        has_files = True
                if has_files:
                    t_set.add(t)

            if not file_index:
                return None

            times = sorted(t_set)
            z_levels = sorted(z_set)
            channel_names = sorted(c_set)
            n_t, n_fov, n_z, n_c = (
                len(times),
                len(fovs),
                len(z_levels),
                len(channel_names),
            )

            sample = next(
                (
                    p
                    for p in file_index.values()
                    if Path(p).suffix.lower() in TIFF_EXTENSIONS
                ),
                None,
            )
            if sample is None:
                return None
            try:
                with tf.TiffFile(sample) as tif:
                    height, width = tif.pages[0].shape[-2:]
            except Exception as e:
                logger.debug("Failed to read sample TIFF: %s", e)
                return None

            luts = {
                i: wavelength_to_colormap(extract_wavelength(c))
                for i, c in enumerate(channel_names)
            }

            @lru_cache(maxsize=128)
            def load_plane(t, region, fov, z, channel):
                filepath = file_index.get((t, region, fov, z, channel))
                if not filepath:
                    return np.zeros((height, width), dtype=np.uint16)
                try:
                    ext = Path(filepath).suffix.lower()
                    if ext in TIFF_EXTENSIONS:
                        with tf.TiffFile(filepath) as tif:
                            return tif.pages[0].asarray()
                except Exception as e:
                    logger.debug("Failed to load plane %s: %s", filepath, e)
                return np.zeros((height, width), dtype=np.uint16)

            # Build on-demand loader via map_blocks over a dummy array
            chunks = (
                (1,) * n_t,
                (1,) * n_fov,
                (1,) * n_z,
                (1,) * n_c,
                (height,),
                (width,),
            )

            def _block_loader(block, block_info=None):
                loc = block_info[None]["chunk-location"]
                t_idx, f_idx, z_idx, c_idx = loc[0], loc[1], loc[2], loc[3]
                t = times[t_idx]
                region, fov = fovs[f_idx]["region"], fovs[f_idx]["fov"]
                z = z_levels[z_idx]
                channel = channel_names[c_idx]
                plane = load_plane(t, region, fov, z, channel)
                return plane.reshape(1, 1, 1, 1, height, width)

            dummy = da.zeros(
                (n_t, n_fov, n_z, n_c, height, width), chunks=chunks, dtype=np.uint16
            )
            stacked = da.map_blocks(
                _block_loader, dummy, dtype=np.uint16, chunks=chunks
            )

            # Read acquisition parameters for pixel size and dz
            pixel_size_um, dz_um = read_acquisition_parameters(base_path)

            # Fallback: try reading pixel size from TIFF metadata tags
            if pixel_size_um is None and sample is not None:
                pixel_size_um = read_tiff_pixel_size(sample)

            xarr = xr.DataArray(
                stacked,
                dims=["time", "fov", "z", "channel", "y", "x"],
                # Use actual values for time/z coords, numeric indices for fov/channel.
                # Channel names are stored in attrs and applied via _lut_controllers.
                coords={
                    "time": times,
                    "fov": list(range(n_fov)),
                    "z": z_levels,
                    "channel": list(range(n_c)),
                },
            )
            xarr.attrs["luts"] = luts
            xarr.attrs["channel_names"] = channel_names

            # Store physical pixel sizes (in micrometers)
            if pixel_size_um is not None:
                xarr.attrs["pixel_size_um"] = pixel_size_um
                xarr.attrs["pixel_size_x_um"] = pixel_size_um
                xarr.attrs["pixel_size_y_um"] = pixel_size_um
            if dz_um is not None:
                xarr.attrs["dz_um"] = dz_um
                xarr.attrs["pixel_size_z_um"] = dz_um

            return xarr
        except Exception as e:
            print(f"Single-TIFF load error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _set_ndv_data(self, data: xr.DataArray):
        """Update NDV viewer with lazy array."""
        global _current_voxel_scale

        if not NDV_AVAILABLE or not self.ndv_viewer:
            return

        # Log scale information and set voxel scale for 3D rendering
        pixel_size = data.attrs.get("pixel_size_um")
        dz = data.attrs.get("dz_um")
        if pixel_size is not None or dz is not None:
            scale_info = []
            if pixel_size is not None:
                scale_info.append(f"XY pixel size: {pixel_size:.4f} µm")
            if dz is not None:
                scale_info.append(f"Z step: {dz:.4f} µm")
            print(f"Scale metadata: {', '.join(scale_info)}")

            # Set voxel scale for 3D rendering (Z scaled relative to XY)
            if pixel_size is not None and dz is not None and pixel_size > 0:
                z_scale = dz / pixel_size
                _current_voxel_scale = (1.0, 1.0, z_scale)
                logger.info(f"Voxel aspect ratio (Z/XY): {z_scale:.2f}")
            else:
                _current_voxel_scale = None
        else:
            _current_voxel_scale = None

        luts = data.attrs.get("luts", {})
        channel_axis = data.dims.index("channel") if "channel" in data.dims else None

        # Recreate viewer with proper dimensions
        # Note: 3D button is always enabled - Downsampling3DXarrayWrapper handles
        # large volumes by automatically downsampling them for OpenGL rendering
        old_widget = self.ndv_viewer.widget()
        layout = self.layout()

        self.ndv_viewer = ndv.ArrayViewer(
            data,
            channel_axis=channel_axis,
            channel_mode="composite",
            luts=luts,
            visible_axes=(-2, -1),  # 2D display (y, x), sliders for rest
        )

        # Replace widget
        idx = layout.indexOf(old_widget)
        layout.removeWidget(old_widget)
        old_widget.deleteLater()
        layout.insertWidget(idx, self.ndv_viewer.widget(), 1)

        # Update channel labels after viewer is ready.
        self._initiate_channel_label_update()

    def _initiate_channel_label_update(self):
        """Start the channel label update retry mechanism.

        Increments generation to cancel any pending retries from previous loads,
        then schedules retry attempts until NDV viewer is ready.
        """
        self._channel_label_generation += 1
        self._pending_channel_label_retries = CHANNEL_LABEL_UPDATE_MAX_RETRIES
        self._schedule_channel_label_update(self._channel_label_generation)

    def _schedule_channel_label_update(self, generation: int):
        """Retry updating channel labels until the NDV viewer is ready or we time out."""
        # Check if this callback is from a stale generation (viewer was replaced)
        if self._channel_label_generation != generation:
            return

        if not self.ndv_viewer or self._xarray_data is None:
            return

        remaining = self._pending_channel_label_retries
        if remaining <= 0:
            logger.warning(
                "Channel label update timed out - labels may show numeric indices"
            )
            return

        # Check if _lut_controllers is available (indicates viewer is ready).
        # Note: _lut_controllers is a private API that may change in future ndv versions;
        # at the time of writing there is no stable public API for this behavior in ndv.
        # If removed or renamed, this retry loop will timeout gracefully and channel
        # labels will not be updated, falling back to numeric indices in the UI.
        controllers = getattr(self.ndv_viewer, "_lut_controllers", None)
        if controllers:
            self._update_channel_labels()
            return

        # Not ready yet; schedule another check
        self._pending_channel_label_retries = remaining - 1
        QTimer.singleShot(
            CHANNEL_LABEL_UPDATE_RETRY_DELAY_MS,
            lambda: self._schedule_channel_label_update(generation),
        )

    def _update_channel_labels(self):
        """Manually update channel labels in the NDV viewer.

        This uses ndv's private _lut_controllers API to set display names.
        The approach is fragile and may break with future ndv updates.
        """
        if not self.ndv_viewer or self._xarray_data is None:
            return

        channel_names = self._xarray_data.attrs.get("channel_names", [])
        if not channel_names:
            return

        try:
            controllers = getattr(self.ndv_viewer, "_lut_controllers", None)
            if not controllers:
                return

            updated_names = []
            for i, name in enumerate(channel_names):
                if i in controllers:
                    controller = controllers[i]
                    controller.key = name
                    if hasattr(controller, "synchronize"):
                        # Propagate the updated key to the NDV UI so the channel
                        # label is displayed in the LUT controls.
                        controller.synchronize()
                    else:
                        logger.warning(
                            "LUT controller at index %d has no 'synchronize' method; "
                            "channel label '%s' may not appear in UI",
                            i,
                            name,
                        )
                    updated_names.append(name)
            logger.debug(
                "Updated %d channel labels: %s", len(updated_names), updated_names
            )
        except Exception as e:
            logger.debug("Failed to update channel labels: %s", e)


class LightweightMainWindow(QMainWindow):
    """Main window with dark theme."""

    viewer: LightweightViewer

    def __init__(self, dataset_path: str):
        super().__init__()
        self.setWindowTitle(f"NDViewer Lightweight - {Path(dataset_path).name}")
        self.setGeometry(100, 100, 720, 540)  # 4:3 aspect, smaller
        self._set_dark_theme()

        self.viewer = LightweightViewer(dataset_path)
        self.setCentralWidget(self.viewer)

    def _set_dark_theme(self):
        _apply_dark_theme(self)

    def closeEvent(self, event):
        """Ensure viewer cleanup when window closes."""
        self.viewer.close()
        super().closeEvent(event)


def main(dataset_path: str = None):
    """Launch lightweight viewer."""
    import sys

    app = QApplication(sys.argv)

    if dataset_path:
        # Direct launch with dataset
        window = LightweightMainWindow(dataset_path)
        window.show()
    else:
        # Show launcher window first
        launcher = LauncherWindow()
        launcher.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
