"""
Lightweight NDV-based viewer

Supports: OME-TIFF and single-TIFF acquisitions with lazy loading via dask.
Lazy loading enables fast initial display by only reading image planes on-demand.
"""

import json
import logging
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPalette, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

# Try to import QIconifyIcon for NDV-style play buttons
try:
    from superqt.iconify import QIconifyIcon

    ICONIFY_AVAILABLE = True
except ImportError:
    ICONIFY_AVAILABLE = False

# Try to import QLabeledSlider from superqt (same as NDV uses)
try:
    from superqt import QLabeledSlider

    SUPERQT_AVAILABLE = True
except ImportError:
    from PyQt5.QtWidgets import QSlider

    SUPERQT_AVAILABLE = False

# NDV slider style (matches NDV's internal sliders)
NDV_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 100, 100, 0.25),
        stop:1 rgba(100, 100, 100, 0.1)
    );
}
QLabel { font-size: 12px; }
SliderLabel { font-size: 10px; }
"""

if TYPE_CHECKING:
    import xarray as xr

import tensorstore as ts

# Constants
TIFF_EXTENSIONS = {".tif", ".tiff"}
LIVE_REFRESH_INTERVAL_MS = 750
SLIDER_PLAY_INTERVAL_MS = 100  # Animation interval for play buttons
ZARR_LOAD_DEBOUNCE_MS = 200  # Debounce interval for zarr frame loading
PLANE_CACHE_MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256MB for z-stack plane cache

# Play button style (matches NDV's PlayButton)
PLAY_BUTTON_STYLE = "QPushButton {border: none; padding: 0; margin: 0;}"


def _create_play_button(parent=None) -> QPushButton:
    """Create a play button matching NDV's style."""
    if ICONIFY_AVAILABLE:
        icn = QIconifyIcon("bi:play-fill", color="#888888")
        icn.addKey("bi:pause-fill", state=QIconifyIcon.State.On, color="#4580DD")
        btn = QPushButton(icn, "", parent)
        btn.setIconSize(QSize(16, 16))
    else:
        btn = QPushButton("▶", parent)
    btn.setCheckable(True)
    btn.setFixedSize(18, 18)
    btn.setStyleSheet(PLAY_BUTTON_STYLE)
    return btn


logger = logging.getLogger(__name__)


class MemoryBoundedLRUCache:
    """Thread-safe LRU cache with memory-based size limit.

    Evicts least-recently-used entries when memory limit is exceeded.
    Designed for caching large numpy arrays (image planes).

    Thread safety is required because dask workers may load planes concurrently.
    """

    def __init__(self, max_memory_bytes: int):
        self._max_memory = max_memory_bytes
        self._current_memory = 0
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[np.ndarray]:
        """Get item from cache, marking it as recently used."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: tuple, value: np.ndarray) -> None:
        """Add item to cache, evicting LRU entries if needed."""
        item_size = value.nbytes

        # Don't cache if single item exceeds limit
        if item_size > self._max_memory:
            logger.debug(
                "Cannot cache item (size %d bytes exceeds max %d bytes): key=%s",
                item_size,
                self._max_memory,
                key,
            )
            return

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._current_memory -= self._cache[key].nbytes
                del self._cache[key]

            # Evict LRU entries until we have room
            while self._current_memory + item_size > self._max_memory and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._current_memory -= oldest_value.nbytes

            self._cache[key] = value
            self._current_memory += item_size

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def invalidate(self, key: tuple) -> bool:
        """Remove a specific entry from cache if present.

        Returns True if entry was removed, False if not found.
        """
        with self._lock:
            if key in self._cache:
                self._current_memory -= self._cache[key].nbytes
                del self._cache[key]
                return True
            return False

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


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
                # Use try/except to handle frozen vispy objects (e.g., napari's Volume)
                global _current_voxel_scale
                try:
                    self._voxel_scale = _current_voxel_scale
                except AttributeError:
                    # Object is frozen (e.g., napari), skip the patch
                    pass

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
    from functools import lru_cache

    import dask.array as da
    import tifffile as tf
    import xarray as xr
    from scipy.ndimage import zoom as ndimage_zoom

    LAZY_LOADING_AVAILABLE = True
except ImportError as e:
    import warnings

    warnings.warn(
        f"Lazy loading disabled: missing dependency '{e.name}'. "
        f"Install it with: pip install {e.name}",
        stacklevel=1,
    )
    LAZY_LOADING_AVAILABLE = False

# OpenGL 3D texture size limit (conservative estimate for most GPUs)
MAX_3D_TEXTURE_SIZE = 2048

# Channel label update retry configuration
CHANNEL_LABEL_UPDATE_MAX_RETRIES = 20
CHANNEL_LABEL_UPDATE_RETRY_DELAY_MS = 100

# Register custom DataWrapper for automatic 3D downsampling
if NDV_AVAILABLE and LAZY_LOADING_AVAILABLE:
    from collections.abc import Mapping

    from ndv.models._data_wrapper import XarrayWrapper

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

                    from OpenGL.GL import GL_MAX_3D_TEXTURE_SIZE, glGetIntegerv

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
            logger.debug(
                "Downsampling3DXarrayWrapper.isel: index=%s, data.shape=%s, dims=%s",
                index, data.shape, self._data.dims,
            )

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
    """Detect dataset format: zarr_v3, ome_tiff, or single_tiff.

    Zarr v3 is detected by:
    - plate.zarr or plate.ome.zarr directory (HCS format)
    - zarr/ directory with .zarr or .ome.zarr subdirectories
    - base_path itself being a .zarr/.ome.zarr directory with zarr.json

    OME-TIFF is detected by .ome.tif* files.
    Falls back to single_tiff if neither is detected.
    """
    # Check for zarr v3 formats
    # 1. HCS plate format: plate.zarr/ or plate.ome.zarr/
    if (base_path / "plate.zarr").exists() or (base_path / "plate.ome.zarr").exists():
        return "zarr_v3"

    # 2. Non-HCS: zarr/ directory with .zarr or .ome.zarr subdirs
    zarr_dir = base_path / "zarr"
    if zarr_dir.exists():
        for region_dir in zarr_dir.iterdir():
            if region_dir.is_dir() and not region_dir.name.startswith("."):
                # Check for acquisition.zarr or fov_*.zarr (old format)
                if (region_dir / "acquisition.zarr").exists():
                    return "zarr_v3"
                # Check for .zarr or .ome.zarr subdirectories
                for d in region_dir.iterdir():
                    if d.is_dir() and (
                        d.suffix == ".zarr" or d.name.endswith(".ome.zarr")
                    ):
                        return "zarr_v3"

    # 3. Direct .zarr or .ome.zarr directory
    if base_path.suffix == ".zarr" or base_path.name.endswith(".ome.zarr"):
        if (base_path / "zarr.json").exists():
            return "zarr_v3"

    # Check for OME-TIFF
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


def detect_zarr_version(path: Path) -> Optional[int]:
    """Detect if path is zarr v2 (.zgroup/.zarray) or v3 (zarr.json).

    Args:
        path: Path to zarr store

    Returns:
        2 for zarr v2, 3 for zarr v3, None if unknown
    """
    if (path / "zarr.json").exists():
        return 3
    if (path / ".zgroup").exists() or (path / ".zarray").exists():
        return 2
    # Check for array subdirectory (e.g., "0")
    arr_path = path / "0"
    if arr_path.is_dir():
        if (arr_path / "zarr.json").exists():
            return 3
        if (arr_path / ".zarray").exists():
            return 2
    return None


class _TensorStoreArrayWrapper:
    """Thin wrapper around a tensorstore array exposing numpy-compatible dtype.

    Dask's ``from_array`` calls ``np.dtype(arr.dtype)`` internally, which fails
    with tensorstore's own dtype objects.  This wrapper delegates all indexing
    to the underlying tensorstore array while reporting a plain numpy dtype so
    that dask can build its task graph without errors.
    """

    def __init__(self, ts_array):
        self._arr = ts_array
        self.shape = tuple(ts_array.shape)
        self.dtype = np.dtype(ts_array.dtype.numpy_dtype)
        self.ndim = len(ts_array.shape)

    def __getitem__(self, idx):
        return np.asarray(self._arr[idx].read().result())


def open_zarr_tensorstore(path: Path, array_path: str = "0") -> Optional[Any]:
    """Open a zarr store using tensorstore, auto-detecting v2/v3 format.

    Args:
        path: Path to zarr store
        array_path: Path to array within store (default "0" for OME-NGFF)

    Returns:
        TensorStore array object, or None if failed
    """
    version = detect_zarr_version(path)
    if version is None:
        logger.warning(f"Could not detect zarr version for {path}")
        return None

    # Build tensorstore spec
    driver = "zarr3" if version == 3 else "zarr"
    full_path = path / array_path if array_path else path

    spec = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": str(full_path)},
        # Revalidate metadata on each read (default "open" only checks at open time)
        "recheck_cached_metadata": True,
    }

    try:
        store = ts.open(spec, read=True).result()
        return store
    except Exception as e:
        logger.debug(f"Failed to open zarr store with tensorstore: {e}")
        return None


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


def hex_to_colormap(hex_color: str) -> str:
    """Convert hex RGB color to nearest NDV colormap name.

    Args:
        hex_color: Hex color string, e.g., "#20ADF8" or "20ADF8"

    Returns:
        NDV colormap name: blue, green, yellow, red, magenta, cyan, or gray.
    """
    if not hex_color:
        return "gray"

    # Remove '#' prefix if present
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "gray"

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return "gray"

    # Define reference colors for NDV colormaps (approximate RGB values)
    colormap_refs = {
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "red": (255, 0, 0),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
        "gray": (128, 128, 128),
    }

    # Find nearest colormap by Euclidean distance in RGB space
    min_dist = float("inf")
    best_colormap = "gray"
    for name, (ref_r, ref_g, ref_b) in colormap_refs.items():
        dist = (r - ref_r) ** 2 + (g - ref_g) ** 2 + (b - ref_b) ** 2
        if dist < min_dist:
            min_dist = dist
            best_colormap = name

    return best_colormap


def parse_zarr_v3_metadata(zarr_path: Path) -> dict:
    """Parse OME-NGFF metadata from zarr v3 store.

    Reads metadata from zarr.json -> attributes with ome.multiscales/omero and _squid.

    Extracts metadata from zarr v3 stores:
    - multiscales[0].axes: axis order and names
    - multiscales[0].coordinateTransformations: physical pixel sizes
    - omero.channels: channel names and hex colors
    - _squid: Squid-specific metadata (physical sizes, acquisition_complete)

    Args:
        zarr_path: Path to a .zarr directory

    Returns:
        Dict with keys:
        - axes: List of axis dicts from multiscales
        - pixel_size_um: XY pixel size in micrometers (or None)
        - dz_um: Z step in micrometers (or None)
        - channel_names: List of channel names
        - channel_colors: List of hex color strings
        - acquisition_complete: bool (True if acquisition is finished)
    """
    result = {
        "axes": [],
        "pixel_size_um": None,
        "dz_um": None,
        "channel_names": [],
        "channel_colors": [],
        "acquisition_complete": False,
    }

    # Read metadata from zarr.json
    attrs = None
    zarr_json_path = zarr_path / "zarr.json"

    if zarr_json_path.exists():
        try:
            with open(zarr_json_path, "r") as f:
                zarr_json = json.load(f)
            # Metadata is in zarr.json -> attributes
            attrs = zarr_json.get("attributes", {})
        except Exception as e:
            logger.debug("Failed to read zarr.json: %s", e)

    if not attrs:
        if zarr_json_path.exists():
            logger.warning(
                "zarr.json exists but could not be parsed for %s, using defaults",
                zarr_path,
            )
        return result

    # Handle both old format (multiscales at root) and new format (ome.multiscales)
    ome = attrs.get("ome", {})
    multiscales = ome.get("multiscales") or attrs.get("multiscales", [])
    if multiscales and isinstance(multiscales, list):
        ms = multiscales[0]
        result["axes"] = ms.get("axes", [])

        # Extract physical scales from coordinateTransformations
        coord_transforms = ms.get("coordinateTransformations", [])
        if not coord_transforms:
            # Also check datasets[0].coordinateTransformations
            datasets = ms.get("datasets", [])
            if datasets:
                coord_transforms = datasets[0].get("coordinateTransformations", [])

        for transform in coord_transforms:
            if transform.get("type") == "scale":
                scales = transform.get("scale", [])
                # Map axis names to scale values
                axes = result["axes"]
                for i, ax in enumerate(axes):
                    if i < len(scales):
                        ax_name = ax.get("name", "").lower()
                        ax_unit = ax.get("unit", "").lower()
                        scale_val = scales[i]
                        # Convert to micrometers if needed
                        if ax_unit == "nanometer":
                            scale_val = scale_val / 1000.0
                        elif ax_unit == "millimeter":
                            scale_val = scale_val * 1000.0
                        elif ax_unit == "meter":
                            scale_val = scale_val * 1e6
                        # Assign to appropriate field
                        if ax_name in ("x", "y"):
                            if result["pixel_size_um"] is None:
                                result["pixel_size_um"] = scale_val
                        elif ax_name == "z":
                            result["dz_um"] = scale_val
                break  # Only use first scale transform

    # Handle both old format (omero at root) and new format (ome.omero)
    omero = ome.get("omero") or attrs.get("omero", {})
    channels = omero.get("channels", [])
    for ch in channels:
        name = ch.get("label") or ch.get("name") or ""
        result["channel_names"].append(name)
        # Color can be hex string or integer
        color = ch.get("color", "")
        if isinstance(color, int):
            color = f"{color:06X}"
        result["channel_colors"].append(color)

    # Handle both old format (_squid_metadata) and new format (_squid)
    squid_meta = attrs.get("_squid") or attrs.get("_squid_metadata", {})
    if squid_meta:
        if "pixel_size_um" in squid_meta:
            result["pixel_size_um"] = squid_meta["pixel_size_um"]
        if "z_step_um" in squid_meta:
            result["dz_um"] = squid_meta["z_step_um"]
        result["acquisition_complete"] = squid_meta.get("acquisition_complete", False)

    return result


def discover_zarr_v3_fovs(base_path: Path) -> Tuple[List[Dict], str]:
    """Discover zarr v3 FOV stores within a dataset directory.

    Scans for zarr v3 structures created by Squid (both old and new formats):

    Old format:
    1. HCS plate: plate.zarr/row/col/field/acquisition.zarr
    2. Non-HCS per-FOV: zarr/region/fov_N.zarr
    3. Non-HCS 6D: zarr/region/acquisition.zarr

    New format (PR #474):
    1. HCS plate: plate.ome.zarr/{row}/{col}/{fov}/0 (5D per FOV)
    2. Non-HCS per-FOV: zarr/{region}/fov_{n}.ome.zarr (5D per FOV)
    3. Non-HCS 6D: zarr/{region}/acquisition.zarr (6D with FOV dimension)

    Args:
        base_path: Dataset root directory

    Returns:
        Tuple of:
        - List of dicts: [{"region": str, "fov": int, "path": Path}, ...]
        - Structure type: "hcs_plate", "per_fov", "6d", or "unknown"
    """
    fovs = []

    # Check for HCS plate structure: plate.zarr/ or plate.ome.zarr/
    plate_zarr = None
    if (base_path / "plate.ome.zarr").exists():
        plate_zarr = base_path / "plate.ome.zarr"
    elif (base_path / "plate.zarr").exists():
        plate_zarr = base_path / "plate.zarr"

    if plate_zarr:
        # Scan row/col/field structure
        for row_dir in sorted(plate_zarr.iterdir()):
            if not row_dir.is_dir() or row_dir.name.startswith("."):
                continue
            for col_dir in sorted(row_dir.iterdir()):
                if not col_dir.is_dir() or col_dir.name.startswith("."):
                    continue
                well_id = f"{row_dir.name}{col_dir.name}"
                # Look for field directories (0, 1, 2, ...)
                for field_dir in sorted(col_dir.iterdir()):
                    if not field_dir.is_dir() or not field_dir.name.isdigit():
                        continue
                    field_idx = int(field_dir.name)

                    # New format: {fov}/0 where "0" is the zarr array path
                    array_path = field_dir / "0"
                    if array_path.exists():
                        # The zarr store is the field_dir itself (contains zarr.json and 0/)
                        fovs.append(
                            {"region": well_id, "fov": field_idx, "path": field_dir}
                        )
                        continue

                    # Old format: field/acquisition.zarr
                    acq_zarr = field_dir / "acquisition.zarr"
                    if acq_zarr.exists():
                        fovs.append(
                            {"region": well_id, "fov": field_idx, "path": acq_zarr}
                        )
                        continue

                    # Old format: field/time/acquisition.zarr
                    for time_dir in sorted(field_dir.iterdir()):
                        if not time_dir.is_dir() or not time_dir.name.isdigit():
                            continue
                        acq_zarr = time_dir / "acquisition.zarr"
                        if acq_zarr.exists():
                            fovs.append(
                                {
                                    "region": well_id,
                                    "fov": field_idx,
                                    "path": acq_zarr,
                                }
                            )
                            break  # Use first timepoint's zarr
        if fovs:
            return fovs, "hcs_plate"

    # Check for non-HCS per-FOV: zarr/region/fov_N.zarr or zarr/region/fov_N.ome.zarr
    zarr_dir = base_path / "zarr"
    if zarr_dir.exists():
        for region_dir in sorted(zarr_dir.iterdir()):
            if not region_dir.is_dir() or region_dir.name.startswith("."):
                continue
            region_name = region_dir.name

            # Check for acquisition.zarr (6D single dataset)
            acq_zarr = region_dir / "acquisition.zarr"
            if acq_zarr.exists():
                # This is a 6D single store - return just this one
                fovs.append({"region": region_name, "fov": 0, "path": acq_zarr})
                return fovs, "6d"

            # Look for fov_N.zarr or fov_N.ome.zarr files
            fov_pattern = re.compile(r"fov_(\d+)(?:\.ome)?\.zarr")
            for fov_dir in sorted(region_dir.iterdir()):
                if not fov_dir.is_dir():
                    continue
                m = fov_pattern.match(fov_dir.name)
                if m:
                    fov_idx = int(m.group(1))
                    fovs.append(
                        {"region": region_name, "fov": fov_idx, "path": fov_dir}
                    )
        if fovs:
            return fovs, "per_fov"

    # Check if base_path itself is a .zarr or .ome.zarr directory
    if base_path.suffix == ".zarr" or base_path.name.endswith(".ome.zarr"):
        zarr_json = base_path / "zarr.json"
        if zarr_json.exists():
            fovs.append({"region": "default", "fov": 0, "path": base_path})
            return fovs, "6d"

    return [], "unknown"


def data_structure_changed(
    old_data: Optional["xr.DataArray"], new_data: "xr.DataArray"
) -> bool:
    """Check if data structure changed significantly (requiring full viewer rebuild).

    This is a module-level utility function that detects changes in dimensions,
    dtype, channel count, channel names, or LUT configuration that would require
    rebuilding the NDV viewer rather than just swapping data in-place.

    This function is used by both the LightweightViewer class and unit tests,
    ensuring a single source of truth for the comparison logic.

    Args:
        old_data: Previous dataset state. May be ``None`` if no prior dataset
            exists; when ``None``, the structure is treated as changed.
        new_data: Newly loaded dataset to compare against ``old_data``.

    Returns:
        True if structure changed and viewer needs full rebuild.

    Raises:
        Any exception from xarray attribute access is propagated to the caller.
    """
    if old_data is None:
        return True

    # Check if dims changed
    if old_data.dims != new_data.dims:
        return True

    # Check if dtype changed (may need different contrast limits)
    if old_data.dtype != new_data.dtype:
        return True

    # Check if channel count changed; treat missing "channel" dim as having 0 channels
    if old_data.sizes.get("channel", 0) != new_data.sizes.get("channel", 0):
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
    p.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    p.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    widget.setPalette(p)


class _ScaleBarWidget(QWidget):
    """Custom-painted scale bar overlay for the vispy canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bar_width = 100
        self._text = "100 \u00b5m"
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFixedSize(160, 28)

    def update_bar(self, bar_width_px: int, text: str):
        self._bar_width = bar_width_px
        self._text = text
        self.setFixedSize(max(bar_width_px + 20, 60), 28)
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QFont, QPen, QBrush
        from PyQt5.QtCore import QRectF

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent background
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 4, 4)

        # White bar
        bar_x = (self.width() - self._bar_width) // 2
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(bar_x, 8, bar_x + self._bar_width, 8)
        # End caps
        p.drawLine(bar_x, 5, bar_x, 11)
        p.drawLine(bar_x + self._bar_width, 5, bar_x + self._bar_width, 11)

        # Text
        font = QFont()
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(0, 12, self.width(), 14), Qt.AlignCenter, self._text)

        p.end()


def _set_cephla_icon(window: QMainWindow) -> None:
    """Set the Cephla logo as window icon."""
    try:
        from PyQt5.QtSvg import QSvgRenderer

        logo_path = Path(__file__).parent / "cephla_logo.svg"
        if logo_path.exists():
            renderer = QSvgRenderer(str(logo_path))
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            window.setWindowIcon(QIcon(pixmap))
    except ImportError:
        pass


class LauncherWindow(QMainWindow):
    """Separate launcher window with dropbox for dataset selection."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cephla NDViewer Lightweight - Open Dataset")
        self.setGeometry(100, 100, 400, 300)  # 4:3 aspect, narrower
        self._set_dark_theme()
        _set_cephla_icon(self)

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

        # Subtle branding — logo + text
        brand_widget = QWidget()
        brand_layout = QHBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 0, 0, 4)
        brand_layout.setSpacing(6)
        brand_layout.addStretch()

        logo_path = Path(__file__).parent / "cephla_logo.svg"
        if logo_path.exists():
            try:
                from PyQt5.QtSvg import QSvgRenderer
                logo_label = QLabel()
                renderer = QSvgRenderer(str(logo_path))
                pm = QPixmap(14, 14)
                pm.fill(Qt.transparent)
                p = QPainter(pm)
                p.setOpacity(80 / 255)
                renderer.render(p)
                p.end()
                logo_label.setPixmap(pm)
                brand_layout.addWidget(logo_label)
            except ImportError:
                pass

        brand_text = QLabel("cephla")
        brand_text.setStyleSheet(
            "color: rgba(49, 196, 243, 80); font-size: 10px; "
            "letter-spacing: 3px;"
        )
        brand_layout.addWidget(brand_text)
        brand_layout.addStretch()
        layout.addWidget(brand_widget)

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
    """Minimal NDV-based viewer with external FOV/Time navigation.

    For live acquisition, use the push-based API:
    - start_acquisition() to configure channels, z-levels, dimensions
    - register_image() to register each saved image
    - load_fov() to navigate to a specific position

    For viewing existing datasets, use load_dataset().
    """

    # Signal for thread-safe UI updates from register_image()
    # Signature: (t, fov_idx, _unused1, _unused2) - last two reserved for future use
    _image_registered = pyqtSignal(int, int, int, int)

    # Signal for thread-safe UI updates from notify_zarr_frame()
    # Signature: (t, fov_idx, z, channel_idx)
    _zarr_frame_registered = pyqtSignal(int, int, int, int)

    dataset_path: str
    ndv_viewer: Optional["ndv.ArrayViewer"]
    _xarray_data: Optional["xr.DataArray"]
    _open_handles: List
    _last_sig: Optional[tuple]
    _refresh_timer: Optional[QTimer]

    # Zarr push-based API state
    _zarr_acquisition_active: bool
    _zarr_channel_map: Dict[str, int]

    def __init__(self, dataset_path: str = ""):
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

        # External navigation state (push-based API for live acquisition)
        # _file_index is accessed from both main thread and dask workers, needs lock
        # (t, fov_idx, z, channel) -> (filepath, page_idx)
        self._file_index: Dict[Tuple[int, int, int, str], Tuple[str, int]] = {}
        self._file_index_lock = threading.Lock()
        # LRU cache of open TiffFile handles to avoid re-parsing IFD chains.
        # Each entry is (TiffFile, per-file Lock) so reads to different files
        # proceed in parallel while same-file reads are serialized.
        # Bounded to avoid fd exhaustion with thousands of files.
        self._tiff_handles: OrderedDict = OrderedDict()  # filepath -> (tif, lock)
        self._tiff_handles_lock = threading.Lock()  # protects the dict itself
        self._tiff_handles_max = 128
        self._fov_labels: List[str] = []  # ["A1:0", "A1:1", ...]
        self._channel_names: List[str] = []
        self._z_levels: List[int] = []
        self._luts: Dict[int, Any] = {}  # channel_idx -> colormap
        self._current_fov_idx: int = 0
        self._current_time_idx: int = 0
        self._max_time_idx: int = 0  # Highest t seen (for slider range)
        self._max_fov_per_time: Dict[int, int] = {}  # timepoint -> max FOV index seen
        self._image_height: int = 0
        self._image_width: int = 0
        self._plane_cache = MemoryBoundedLRUCache(PLANE_CACHE_MAX_MEMORY_BYTES)
        self._updating_sliders: bool = False  # Prevent recursive updates
        self._acquisition_active: bool = False  # True during live acquisition
        self._time_play_timer: Optional[QTimer] = None  # Timer for T slider animation
        self._fov_play_timer: Optional[QTimer] = None  # Timer for FOV slider animation
        self._load_debounce_timer: Optional[QTimer] = (
            None  # Debounce for _load_current_fov
        )
        self._load_pending: bool = False  # True if load is scheduled

        # Zarr push-based API state
        self._zarr_acquisition_active: bool = False
        self._zarr_channel_map: Dict[str, int] = {}
        self._zarr_debounce_timer: Optional[QTimer] = None  # Debounce for zarr loads
        self._zarr_load_pending: bool = False
        # For per_fov and hcs structures: separate zarr path per FOV
        self._zarr_fov_paths: List[Path] = []  # [fov0_path, fov1_path, ...]
        self._zarr_fov_stores: Dict[int, Any] = {}  # fov_idx -> zarr store

        # For 6d_regions structure: each region has its own 6D zarr with variable FOV count
        self._zarr_region_paths: List[Path] = []  # [region0.zarr, region1.zarr, ...]
        self._zarr_region_stores: Dict[int, Any] = {}  # region_idx -> zarr store
        self._fovs_per_region: List[int] = []  # [4, 6, 3] - variable per region
        self._region_fov_offsets: List[int] = []  # [0, 4, 10] - cumulative offsets
        self._zarr_6d_regions_mode: bool = False
        self._zarr_written_planes: Set[Tuple] = (
            set()
        )  # Track written planes during live acquisition
        self._zarr_written_planes_lock = (
            threading.Lock()
        )  # Protects _zarr_written_planes
        self._zarr_stores_lock = (
            threading.Lock()
        )  # Protects _zarr_fov_stores and _zarr_region_stores

        # Connect signals for thread-safe updates
        self._image_registered.connect(self._on_image_registered)
        self._zarr_frame_registered.connect(self._on_zarr_frame_registered)

        self._setup_ui()
        if dataset_path:
            self.load_dataset(dataset_path)
        # Note: _setup_live_refresh() removed - using push-based API instead

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

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

        # Create slider container with NDV style
        slider_container = QWidget()
        slider_container.setStyleSheet(NDV_SLIDER_STYLE)
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(5, 2, 5, 2)
        slider_layout.setSpacing(2)

        # Time slider (hidden if only 1 timepoint)
        self._time_container = QWidget()
        t_layout = QHBoxLayout(self._time_container)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(5)
        self._time_label = QLabel("T")
        self._time_label.setFixedWidth(30)
        self._time_play_btn = _create_play_button(self)
        self._time_play_btn.setToolTip("Play/pause automatic timepoint cycling")
        self._time_play_btn.clicked.connect(self._on_time_play_clicked)
        if SUPERQT_AVAILABLE:
            self._time_slider = QLabeledSlider(Qt.Horizontal)
        else:
            self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setMinimum(0)
        self._time_slider.setMaximum(0)
        self._time_slider.setToolTip("Navigate between timepoints")
        self._time_slider.valueChanged.connect(self._on_time_slider_changed)
        self._time_label.setToolTip("Current timepoint index")
        t_layout.addWidget(self._time_play_btn)
        t_layout.addWidget(self._time_label)
        t_layout.addWidget(self._time_slider)
        self._time_container.setVisible(False)  # Hidden by default until max > 0
        slider_layout.addWidget(self._time_container)

        # FOV slider (in a container so we can hide it when there's only one FOV)
        self._fov_slider_container = QWidget()
        fov_layout = QHBoxLayout(self._fov_slider_container)
        fov_layout.setContentsMargins(0, 0, 0, 0)
        fov_layout.setSpacing(5)
        self._fov_label = QLabel("FOV")
        self._fov_label.setMinimumWidth(80)
        self._fov_label.setToolTip("Current field of view (well:index)")
        self._fov_play_btn = _create_play_button(self)
        self._fov_play_btn.setToolTip("Play/pause automatic FOV cycling")
        self._fov_play_btn.clicked.connect(self._on_fov_play_clicked)
        if SUPERQT_AVAILABLE:
            self._fov_slider = QLabeledSlider(Qt.Horizontal)
        else:
            self._fov_slider = QSlider(Qt.Horizontal)
        self._fov_slider.setMinimum(0)
        self._fov_slider.setMaximum(0)
        self._fov_slider.setToolTip("Navigate between fields of view")
        self._fov_slider.valueChanged.connect(self._on_fov_slider_changed)
        fov_layout.addWidget(self._fov_play_btn)
        fov_layout.addWidget(self._fov_label)
        fov_layout.addWidget(self._fov_slider)
        self._fov_slider_container.setVisible(
            False
        )  # Hidden until push API sets FOV labels
        slider_layout.addWidget(self._fov_slider_container)

        layout.addWidget(slider_container)

        self.setLayout(layout)

    def _on_time_slider_changed(self, value: int):
        """Handle time slider change."""
        if self._updating_sliders:
            return
        if value != self._current_time_idx:
            self._current_time_idx = value
            self._time_label.setText(f"T: {value}")

            # Update FOV slider max for this timepoint
            self._updating_sliders = True
            try:
                available_fov_max = self._max_fov_per_time.get(value, 0)
                self._fov_slider.setMaximum(available_fov_max)

                # Clamp current FOV if it exceeds available range
                if self._current_fov_idx > available_fov_max:
                    self._current_fov_idx = available_fov_max
                    self._fov_slider.setValue(available_fov_max)

                # Update FOV label to reflect current FOV after any clamping
                if self._fov_labels and self._current_fov_idx < len(self._fov_labels):
                    self._fov_label.setText(
                        f"FOV: {self._fov_labels[self._current_fov_idx]}"
                    )
                else:
                    self._fov_label.setText(f"FOV: {self._current_fov_idx}")
            finally:
                self._updating_sliders = False

            self._load_current_position()

    def _on_fov_slider_changed(self, value: int):
        """Handle FOV slider change."""
        if self._updating_sliders:
            return
        if value != self._current_fov_idx:
            self._current_fov_idx = value
            # Update FOV label with well:fov format if available
            if self._fov_labels and value < len(self._fov_labels):
                self._fov_label.setText(f"FOV: {self._fov_labels[value]}")
            else:
                self._fov_label.setText(f"FOV: {value}")
            self._load_current_position()

    def _load_current_position(self):
        """Load data for current position, dispatching to appropriate loader.

        Routes to zarr loader if zarr acquisition is active, otherwise to TIFF loader.
        """
        if self.is_zarr_push_mode_active():
            self._load_current_zarr_fov()
        else:
            self._load_current_fov()

    def _update_fov_slider_visibility(self):
        """Show/hide FOV slider based on number of FOVs."""
        n_fovs = len(self._fov_labels) if self._fov_labels else 1
        self._fov_slider_container.setVisible(n_fovs > 1)

    def _on_time_play_clicked(self, checked: bool):
        """Handle time play button click."""
        if checked:
            # Update text for fallback (iconify handles icon automatically)
            if not ICONIFY_AVAILABLE:
                self._time_play_btn.setText("⏸")
            if self._time_play_timer is None:
                self._time_play_timer = QTimer(self)
                self._time_play_timer.timeout.connect(self._time_play_step)
            self._time_play_timer.start(SLIDER_PLAY_INTERVAL_MS)
        else:
            if not ICONIFY_AVAILABLE:
                self._time_play_btn.setText("▶")
            if self._time_play_timer:
                self._time_play_timer.stop()

    def _time_play_step(self):
        """Advance time slider by one step (looping)."""
        max_t = self._time_slider.maximum()
        if max_t <= 0:
            return
        current = self._time_slider.value()
        next_val = (current + 1) % (max_t + 1)
        self._time_slider.setValue(next_val)

    def _on_fov_play_clicked(self, checked: bool):
        """Handle FOV play button click."""
        if checked:
            if not ICONIFY_AVAILABLE:
                self._fov_play_btn.setText("⏸")
            if self._fov_play_timer is None:
                self._fov_play_timer = QTimer(self)
                self._fov_play_timer.timeout.connect(self._fov_play_step)
            self._fov_play_timer.start(SLIDER_PLAY_INTERVAL_MS)
        else:
            if not ICONIFY_AVAILABLE:
                self._fov_play_btn.setText("▶")
            if self._fov_play_timer:
                self._fov_play_timer.stop()

    def _fov_play_step(self):
        """Advance FOV slider by one step (looping)."""
        max_fov = self._fov_slider.maximum()
        if max_fov <= 0:
            return
        current = self._fov_slider.value()
        next_val = (current + 1) % (max_fov + 1)
        self._fov_slider.setValue(next_val)

    def _stop_play_animation(
        self, timer: Optional[QTimer], button: QPushButton
    ) -> None:
        """Stop a play animation and reset the button state."""
        if timer and timer.isActive():
            timer.stop()
            button.setChecked(False)
            if not ICONIFY_AVAILABLE:
                button.setText("▶")

    # ─────────────────────────────────────────────────────────────────────────
    # Push-based API for live acquisition
    # ─────────────────────────────────────────────────────────────────────────

    def start_acquisition(
        self,
        channels: List[str],
        num_z: int,
        height: int,
        width: int,
        fov_labels: List[str],
    ):
        """Configure viewer for a new acquisition.

        Call this at acquisition start before any register_image() calls.
        Sets up LUTs based on channel wavelengths and configures sliders.

        Args:
            channels: Channel names, e.g. ["BF LED matrix full", "Fluorescence 488 nm Ex"]
            num_z: Number of z-levels
            height: Image height in pixels
            width: Image width in pixels
            fov_labels: FOV labels, e.g. ["A1:0", "A1:1", "A2:0"]
        """
        # Stop any running play animations and pending loads
        self._stop_play_animation(self._time_play_timer, self._time_play_btn)
        self._stop_play_animation(self._fov_play_timer, self._fov_play_btn)
        if self._load_debounce_timer and self._load_debounce_timer.isActive():
            self._load_debounce_timer.stop()
        self._load_pending = False

        # Clear previous state
        with self._file_index_lock:
            self._file_index.clear()
        self._close_tiff_handle_cache()
        self._plane_cache.clear()
        self._max_fov_per_time.clear()

        # Store configuration
        self._channel_names = list(channels)
        self._z_levels = list(range(num_z))
        self._image_height = height
        self._image_width = width
        self._fov_labels = list(fov_labels)

        # Set up LUTs based on channel wavelengths
        self._luts = {
            i: wavelength_to_colormap(extract_wavelength(c))
            for i, c in enumerate(self._channel_names)
        }

        # Reset navigation state
        self._current_fov_idx = 0
        self._current_time_idx = 0
        self._max_time_idx = 0
        self._acquisition_active = True

        # Update sliders
        self._updating_sliders = True
        try:
            self._time_slider.setMaximum(0)
            self._time_slider.setValue(0)
            self._time_label.setText("T: 0")

            self._fov_slider.setMaximum(0)  # Start at 0, grows as FOVs are acquired
            self._fov_slider.setValue(0)
            if fov_labels:
                self._fov_label.setText(f"FOV: {fov_labels[0]}")
            else:
                self._fov_label.setText("FOV: -")
        finally:
            self._updating_sliders = False

        # Rebuild NDV viewer with channel configuration
        self._rebuild_viewer_for_acquisition()

        # Show/hide FOV slider based on number of FOVs
        self._update_fov_slider_visibility()

        logger.info(
            f"NDViewer: Started acquisition with {len(channels)} channels, "
            f"{num_z} z-levels, {len(fov_labels)} FOVs"
        )

    def _rebuild_viewer_for_acquisition(self):
        """Rebuild the NDV viewer for the current acquisition configuration."""
        if not NDV_AVAILABLE or not self.ndv_viewer:
            return

        # Create placeholder array with correct shape: z_level × channel × y × x
        n_z = len(self._z_levels) if self._z_levels else 1
        n_c = len(self._channel_names) if self._channel_names else 1
        h = self._image_height if self._image_height > 0 else 100
        w = self._image_width if self._image_width > 0 else 100

        placeholder = np.zeros((n_z, n_c, h, w), dtype=np.uint16)

        import xarray as xr

        xarr = xr.DataArray(
            placeholder,
            dims=["z_level", "channel", "y", "x"],
            coords={
                "z_level": self._z_levels if self._z_levels else [0],
                "channel": list(range(n_c)),
            },
        )
        xarr.attrs["luts"] = self._luts
        xarr.attrs["channel_names"] = self._channel_names

        self._xarray_data = xarr
        self._set_ndv_data(xarr)

    def register_image(
        self,
        t: int,
        fov_idx: int,
        z: int,
        channel: str,
        filepath: str,
        page_idx: int = 0,
    ):
        """Register a newly saved image file.

        Thread-safe: can be called from worker thread.
        Updates file index and emits signal for GUI update.

        Args:
            t: Timepoint index
            fov_idx: FOV index (0-based)
            z: Z-level index
            channel: Channel name
            filepath: Path to the saved TIFF file
            page_idx: Page index within the TIFF file (default 0).
                For OME-TIFF stacks that store multiple planes per file,
                specify which page to read.
        """
        if page_idx < 0:
            raise ValueError(f"page_idx must be >= 0, got {page_idx}")
        # Update file index (protected by lock for dask worker thread safety)
        with self._file_index_lock:
            self._file_index[(t, fov_idx, z, channel)] = (filepath, page_idx)

        # Emit signal with raw indices - main thread computes max values
        # to avoid race condition on _max_time_idx
        try:
            self._image_registered.emit(t, fov_idx, 0, 0)
        except RuntimeError as e:
            # Qt object deleted - viewer was closed during acquisition
            logger.warning(
                "Could not emit image_registered signal (viewer may be closed): %s", e
            )

    def _on_image_registered(self, t: int, fov_idx: int, _unused1: int, _unused2: int):
        """Handle image registration signal (runs on main thread).

        Updates slider ranges and schedules debounced FOV load if needed.

        Note: _unused1/_unused2 are placeholder parameters kept for signal
        compatibility; max values are computed here from the current tracking
        state (_max_time_idx, _max_fov_per_time), not passed via signal.
        """
        try:
            # Update per-timepoint max FOV tracking
            current_max_for_t = self._max_fov_per_time.get(t, -1)
            if fov_idx > current_max_for_t:
                self._max_fov_per_time[t] = fov_idx

            # Compute max time
            new_max_t = max(self._max_time_idx, t)

            self._updating_sliders = True
            try:
                # Update T slider if needed
                if new_max_t > self._max_time_idx:
                    self._max_time_idx = new_max_t
                    self._time_slider.setMaximum(new_max_t)

                # Show T slider if we have multiple timepoints
                if new_max_t > 0:
                    self._time_container.setVisible(True)

                # Update FOV slider max for CURRENT timepoint only
                if t == self._current_time_idx:
                    current_fov_max = self._fov_slider.maximum()
                    available_fov_max = self._max_fov_per_time.get(t, 0)
                    if available_fov_max > current_fov_max:
                        self._fov_slider.setMaximum(available_fov_max)
            finally:
                self._updating_sliders = False

            # Schedule debounced load if this image is for the current FOV
            if t == self._current_time_idx and fov_idx == self._current_fov_idx:
                self._schedule_debounced_load()
        except Exception as e:
            logger.error("Error in _on_image_registered: %s", e, exc_info=True)

    def _schedule_debounced_load(self):
        """Schedule a debounced load of the current FOV.

        Coalesces rapid image registrations into a single load every 200ms.
        This prevents overwhelming the main thread during fast acquisitions.
        """
        # Mark that a load is pending
        self._load_pending = True

        # Create timer if needed
        if self._load_debounce_timer is None:
            self._load_debounce_timer = QTimer(self)
            self._load_debounce_timer.setSingleShot(True)
            self._load_debounce_timer.timeout.connect(self._execute_debounced_load)

        # If timer not running, start it; otherwise the existing timer will handle it
        if not self._load_debounce_timer.isActive():
            self._load_debounce_timer.start(200)  # 200ms debounce

    def _execute_debounced_load(self):
        """Execute the debounced FOV load."""
        if self._load_pending:
            self._load_pending = False
            self._load_current_fov()

    def load_fov(self, fov: int, t: Optional[int] = None, z: Optional[int] = None):
        """Load and display a specific FOV.

        Args:
            fov: FOV index to display
            t: Timepoint index (None = use current)
            z: Z-level index (None = use current, not used for NDV internal z)

        Only updates data, LUTs remain unchanged.
        """
        if t is not None:
            self._current_time_idx = t
        if fov != self._current_fov_idx:
            self._current_fov_idx = fov

        # Update sliders to reflect new position
        self._updating_sliders = True
        try:
            self._time_slider.setValue(self._current_time_idx)
            self._time_label.setText(f"T: {self._current_time_idx}")
            self._fov_slider.setValue(self._current_fov_idx)
            if self._fov_labels and self._current_fov_idx < len(self._fov_labels):
                self._fov_label.setText(
                    f"FOV: {self._fov_labels[self._current_fov_idx]}"
                )
            else:
                self._fov_label.setText(f"FOV: {self._current_fov_idx}")
        finally:
            self._updating_sliders = False

        self._load_current_position()

    def go_to_well_fov(self, well_id: str, fov_index: int) -> bool:
        """Navigate to a specific well and FOV (push-based API).

        Maps (well_id, fov_index) to flat FOV index using _fov_labels.
        Labels are in format "A1:0", "A1:1", "A2:0", etc.

        Args:
            well_id: Well identifier (e.g., "A1", "B2")
            fov_index: FOV index within the well

        Returns:
            True if navigation succeeded, False if FOV not found.
        """
        if not self._fov_labels:
            logger.debug("go_to_well_fov: no FOV labels available")
            return False

        # Find the flat index for this well:fov combination
        target_label = f"{well_id}:{fov_index}"
        try:
            flat_idx = self._fov_labels.index(target_label)
        except ValueError:
            logger.debug(
                f"go_to_well_fov: label '{target_label}' not found in {self._fov_labels}"
            )
            return False

        self.load_fov(flat_idx)
        logger.info(
            f"go_to_well_fov: navigated to {target_label} (flat_idx={flat_idx})"
        )
        return True

    def is_push_mode_active(self) -> bool:
        """Check if push-based mode is active (has FOV labels configured or zarr mode)."""
        return bool(self._fov_labels) or self._zarr_acquisition_active

    def _load_single_plane(
        self, t: int, fov_idx: int, z: int, channel: str
    ) -> np.ndarray:
        """Load a single image plane from cache or disk.

        Args:
            t: Timepoint index
            fov_idx: FOV index
            z: Z-level value
            channel: Channel name

        Returns:
            Image plane as numpy array, or zeros if not available.
        """
        cache_key = (t, fov_idx, z, channel)

        # Check cache first
        cached_plane = self._plane_cache.get(cache_key)
        if cached_plane is not None:
            return cached_plane

        # Load from file (lock protects concurrent access from dask workers)
        with self._file_index_lock:
            entry = self._file_index.get(cache_key)

        if entry is None:
            # File not yet registered - expected during acquisition, not an error
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

        filepath, page_idx = entry

        if not LAZY_LOADING_AVAILABLE:
            logger.error("tifffile not available for loading image planes")
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

        try:
            # Look up or create a cached (TiffFile, Lock) entry.
            # Global lock is held only for dict bookkeeping; the per-file
            # lock serializes reads to the same file while allowing parallel
            # reads across different files.
            evicted_entries = []
            with self._tiff_handles_lock:
                entry = self._tiff_handles.get(filepath)
                if entry is not None:
                    tif, file_lock = entry
                    self._tiff_handles.move_to_end(filepath)
                else:
                    tif = tf.TiffFile(filepath)
                    file_lock = threading.Lock()
                    self._tiff_handles[filepath] = (tif, file_lock)
                    # Evict LRU handles if over limit
                    while len(self._tiff_handles) > self._tiff_handles_max:
                        _, evicted = self._tiff_handles.popitem(last=False)
                        evicted_entries.append(evicted)
            # Close evicted handles.  Safe because LRU eviction only removes
            # the oldest entry, which the current thread just moved away from
            # (move_to_end).  For another thread to hold a reference to the
            # evicted entry, 128+ new files would need to open in the
            # microseconds between its lookup and file_lock acquire.
            for old_tif, old_lock in evicted_entries:
                with old_lock:
                    self._close_tiff_handles([old_tif])
            # Read page under per-file lock
            with file_lock:
                plane = tif.pages[page_idx].asarray()
            self._plane_cache.put(cache_key, plane)
            return plane
        except FileNotFoundError:
            logger.warning("Image file not found (may have been deleted): %s", filepath)
        except IndexError:
            # Evict stale entry so next lookup re-opens the file and sees
            # newly appended pages.  Don't close the handle here — another
            # thread may still hold a reference and be about to read.  The
            # handle will be closed at end_acquisition() / closeEvent().
            with self._tiff_handles_lock:
                self._tiff_handles.pop(filepath, None)
            logger.warning(
                "Page %d not available in %s (file may still be writing)",
                page_idx,
                filepath,
            )
        except PermissionError as e:
            logger.error("Permission denied reading image %s: %s", filepath, e)
        except Exception as e:
            logger.error(
                "Failed to load image plane %s page %d (t=%d, fov=%d, z=%d, ch=%s): %s",
                filepath,
                page_idx,
                t,
                fov_idx,
                z,
                channel,
                e,
                exc_info=True,
            )

        # Return zeros on error - user sees black image
        return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

    def _load_current_fov(self):
        """Load and display data for the current FOV position.

        Creates a lazy dask array that only loads planes when NDV requests them.
        This avoids loading all z-planes when only one is displayed.
        """
        # Check if we have data configuration (set by start_acquisition)
        if not self._channel_names or not self._z_levels:
            return
        if self._image_height == 0 or self._image_width == 0:
            return
        # Check if we have any registered files
        with self._file_index_lock:
            if not self._file_index:
                return

        t = self._current_time_idx
        fov_idx = self._current_fov_idx
        h, w = self._image_height, self._image_width

        # Create lazy dask array - planes only load when accessed
        import dask
        import dask.array as da

        delayed_planes = []
        for z in self._z_levels:
            channel_planes = []
            for channel in self._channel_names:
                # Create delayed load - no disk I/O happens here
                delayed_load = dask.delayed(self._load_single_plane)(
                    t, fov_idx, z, channel
                )
                da_plane = da.from_delayed(delayed_load, shape=(h, w), dtype=np.uint16)
                channel_planes.append(da_plane)
            # Stack channels: (n_c, h, w)
            delayed_planes.append(da.stack(channel_planes))
        # Stack z-levels: (n_z, n_c, h, w)
        data = da.stack(delayed_planes)

        # Update NDV viewer data without rebuilding (preserves LUTs)
        self._update_ndv_data(data)

    def _update_ndv_data(self, data):
        """Update NDV viewer with new data array, preserving LUTs.

        Args:
            data: numpy or dask array of shape (z_level, channel, y, x).
                  Dask arrays enable lazy loading - planes only load when displayed.
        """
        if not NDV_AVAILABLE or not self.ndv_viewer:
            return

        import xarray as xr

        xarr = xr.DataArray(
            data,
            dims=["z_level", "channel", "y", "x"],
            coords={
                "z_level": self._z_levels,
                "channel": list(range(len(self._channel_names))),
            },
        )
        xarr.attrs["luts"] = self._luts
        xarr.attrs["channel_names"] = self._channel_names

        self._xarray_data = xarr

        # Try in-place update to avoid flickering
        if not self._try_inplace_ndv_update(xarr):
            # Fallback: full rebuild (shouldn't happen often)
            self._set_ndv_data(xarr)

    def end_acquisition(self):
        """Mark acquisition as ended.

        Call this when acquisition completes. The viewer remains in push mode
        (is_push_mode_active() returns True) so navigation via go_to_well_fov()
        continues to work for browsing the acquired data.

        FOV labels are preserved to enable navigation. They are only cleared
        when a new acquisition starts via start_acquisition().
        """
        # Stop any pending debounced load from previous acquisition
        if self._load_debounce_timer and self._load_debounce_timer.isActive():
            self._load_debounce_timer.stop()
        self._load_pending = False

        self._acquisition_active = False
        # NOTE: _fov_labels is NOT cleared here - navigation must still work
        # after acquisition ends. Labels are cleared in start_acquisition().
        self._close_tiff_handle_cache()
        logger.info("NDViewer: Acquisition ended")

    # ─────────────────────────────────────────────────────────────────────────
    # Push-based Zarr API for live acquisition
    # ─────────────────────────────────────────────────────────────────────────

    def start_zarr_acquisition(
        self,
        fov_paths: List[str],
        channels: List[str],
        num_z: int,
        fov_labels: List[str],
        height: int,
        width: int,
    ):
        """Configure viewer for per-FOV zarr-based live acquisition.

        Call this at acquisition start before any notify_zarr_frame() calls.
        Sets up the viewer with channel configuration. Each FOV has its own
        5D zarr store with shape (T, C, Z, Y, X).

        For 6D zarr structures (FOV, T, C, Z, Y, X), use start_zarr_acquisition_6d() instead.

        Args:
            fov_paths: List of zarr paths, one per FOV (e.g., ["fov_0.zarr", "fov_1.zarr"])
            channels: Channel names, e.g. ["DAPI", "GFP", "RFP"]
            num_z: Number of z-levels
            fov_labels: FOV labels, e.g. ["A1:0", "A1:1", "A2:0"]
            height: Image height in pixels
            width: Image width in pixels
        """
        # Validate inputs
        if not fov_paths:
            raise ValueError("fov_paths must not be empty")

        # Stop any running animations and pending loads
        self._stop_play_animation(self._time_play_timer, self._time_play_btn)
        self._stop_play_animation(self._fov_play_timer, self._fov_play_btn)
        if self._zarr_debounce_timer and self._zarr_debounce_timer.isActive():
            self._zarr_debounce_timer.stop()
        self._zarr_load_pending = False

        # Close any existing zarr stores
        with self._zarr_stores_lock:
            self._zarr_fov_stores.clear()
            # Also clear 6D regions state (in case switching modes)
            self._zarr_region_stores.clear()
        self._zarr_region_paths = []
        self._fovs_per_region = []
        self._region_fov_offsets = []
        self._zarr_6d_regions_mode = False

        # Clear previous state
        self._plane_cache.clear()
        self._max_fov_per_time.clear()
        with self._zarr_written_planes_lock:
            self._zarr_written_planes.clear()

        # Store configuration
        self._channel_names = list(channels)
        self._z_levels = list(range(num_z))
        self._image_height = height
        self._image_width = width
        self._fov_labels = list(fov_labels)

        # Validate and set per-FOV paths
        if len(fov_paths) != len(fov_labels):
            logger.warning(
                f"fov_paths length ({len(fov_paths)}) does not match "
                f"fov_labels length ({len(fov_labels)}), truncating to shorter"
            )
            min_len = min(len(fov_paths), len(fov_labels))
            fov_paths = list(fov_paths)[:min_len]
            fov_labels = list(fov_labels)[:min_len]
            self._fov_labels = fov_labels
        self._zarr_fov_paths = [Path(p) for p in fov_paths]

        # Build channel name to index map
        self._zarr_channel_map = {name: i for i, name in enumerate(self._channel_names)}

        # Set up LUTs based on channel wavelengths
        self._luts = {
            i: wavelength_to_colormap(extract_wavelength(c))
            for i, c in enumerate(self._channel_names)
        }

        # Note: Zarr stores are opened lazily in _load_zarr_plane to avoid
        # blocking I/O at acquisition start. The store may not exist yet when
        # this method is called since the writer creates it asynchronously.

        # Reset navigation state
        self._current_fov_idx = 0
        self._current_time_idx = 0
        self._max_time_idx = 0
        self._zarr_acquisition_active = True

        # Update sliders
        self._updating_sliders = True
        try:
            self._time_slider.setMaximum(0)
            self._time_slider.setValue(0)
            self._time_label.setText("T: 0")

            self._fov_slider.setMaximum(0)
            self._fov_slider.setValue(0)
            if fov_labels:
                self._fov_label.setText(f"FOV: {fov_labels[0]}")
            else:
                self._fov_label.setText("FOV: -")
        finally:
            self._updating_sliders = False

        # Rebuild NDV viewer with channel configuration
        self._rebuild_viewer_for_acquisition()

        # Show/hide FOV slider based on number of FOVs
        self._update_fov_slider_visibility()

        logger.info(
            f"NDViewer: Started per-FOV zarr acquisition with {len(channels)} channels, "
            f"{num_z} z-levels, {len(self._zarr_fov_paths)} FOVs"
        )

    def start_zarr_acquisition_6d(
        self,
        region_paths: List[str],
        channels: List[str],
        num_z: int,
        fovs_per_region: List[int],
        height: int,
        width: int,
        region_labels: Optional[List[str]] = None,
    ):
        """Configure viewer for multi-region 6D zarr-based live acquisition.

        Each region has its own zarr file with shape (FOV, T, C, Z, Y, X).
        FOVs are flattened across regions with labels showing "region:fov" format.

        Args:
            region_paths: Paths to zarr stores, one per region
            channels: Channel names, e.g. ["DAPI", "GFP", "RFP"]
            num_z: Number of z-levels
            fovs_per_region: Number of FOVs per region, e.g. [4, 6, 3]
            height: Image height in pixels
            width: Image width in pixels
            region_labels: Optional region labels (auto-generated if not provided)
        """
        # Stop any running animations and pending loads
        self._stop_play_animation(self._time_play_timer, self._time_play_btn)
        self._stop_play_animation(self._fov_play_timer, self._fov_play_btn)
        if self._zarr_debounce_timer and self._zarr_debounce_timer.isActive():
            self._zarr_debounce_timer.stop()
        self._zarr_load_pending = False

        # Close any existing zarr stores
        with self._zarr_stores_lock:
            self._zarr_fov_stores.clear()
            self._zarr_region_stores.clear()

        # Clear previous state
        self._plane_cache.clear()
        self._max_fov_per_time.clear()
        with self._zarr_written_planes_lock:
            self._zarr_written_planes.clear()

        # Validate inputs
        if not region_paths:
            raise ValueError("region_paths must not be empty")
        if not fovs_per_region or not all(n > 0 for n in fovs_per_region):
            raise ValueError("fovs_per_region must be non-empty with positive values")
        if len(region_paths) != len(fovs_per_region):
            raise ValueError(
                f"region_paths length ({len(region_paths)}) must match "
                f"fovs_per_region length ({len(fovs_per_region)})"
            )

        # Store configuration
        self._channel_names = list(channels)
        self._z_levels = list(range(num_z))
        self._image_height = height
        self._image_width = width

        # Store region-specific state
        self._zarr_region_paths = [Path(p) for p in region_paths]
        self._fovs_per_region = list(fovs_per_region)

        # Compute cumulative FOV offsets for global→local conversion
        # Example: fovs_per_region=[4, 6, 3] → offsets=[0, 4, 10]
        self._region_fov_offsets = []
        offset = 0
        for n_fov in fovs_per_region:
            self._region_fov_offsets.append(offset)
            offset += n_fov

        # Generate region labels if not provided
        if region_labels is None:
            region_labels = [f"region_{i}" for i in range(len(region_paths))]

        # Generate flattened FOV labels: ["region_0:0", "region_0:1", ..., "region_1:0", ...]
        self._fov_labels = []
        for region_idx, (region_label, n_fov) in enumerate(
            zip(region_labels, fovs_per_region)
        ):
            for fov_in_region in range(n_fov):
                self._fov_labels.append(f"{region_label}:{fov_in_region}")
        logger.info(
            f"6D regions mode: labels={self._fov_labels[:5]}..., paths={len(self._zarr_region_paths)}, fovs_per_region={self._fovs_per_region}, offsets={self._region_fov_offsets}"
        )

        # Build channel name to index map
        self._zarr_channel_map = {name: i for i, name in enumerate(self._channel_names)}

        # Set up LUTs based on channel wavelengths
        self._luts = {
            i: wavelength_to_colormap(extract_wavelength(c))
            for i, c in enumerate(self._channel_names)
        }

        # Clear per-FOV state (not used in 6d_regions mode)
        self._zarr_fov_paths = []

        # Enable 6d_regions mode
        self._zarr_6d_regions_mode = True

        # Reset navigation state
        self._current_fov_idx = 0
        self._current_time_idx = 0
        self._max_time_idx = 0
        self._zarr_acquisition_active = True

        # Update sliders - start at 0, grows as FOVs are acquired
        self._updating_sliders = True
        try:
            self._time_slider.setMaximum(0)
            self._time_slider.setValue(0)
            self._time_label.setText("T: 0")

            self._fov_slider.setMaximum(0)  # Start at 0, grows as FOVs are acquired
            self._fov_slider.setValue(0)
            if self._fov_labels:
                self._fov_label.setText(f"FOV: {self._fov_labels[0]}")
            else:
                self._fov_label.setText("FOV: -")
        finally:
            self._updating_sliders = False

        # Rebuild NDV viewer with channel configuration
        self._rebuild_viewer_for_acquisition()

        # Show/hide FOV slider based on number of FOVs
        self._update_fov_slider_visibility()

        logger.info(
            f"NDViewer: Started 6D regions zarr acquisition with {len(channels)} channels, "
            f"{num_z} z-levels, {len(region_paths)} regions, {sum(fovs_per_region)} total FOVs"
        )

    def _global_to_region_fov(self, global_fov_idx: int) -> Tuple[int, int]:
        """Convert global FOV index to (region_idx, local_fov_idx).

        Args:
            global_fov_idx: Global FOV index (0 to total_fovs-1)

        Returns:
            Tuple of (region_idx, local_fov_idx within that region)
        """
        total_fovs = sum(self._fovs_per_region)
        for region_idx, offset in enumerate(self._region_fov_offsets):
            next_offset = (
                self._region_fov_offsets[region_idx + 1]
                if region_idx + 1 < len(self._region_fov_offsets)
                else total_fovs
            )
            if offset <= global_fov_idx < next_offset:
                return region_idx, global_fov_idx - offset
        return 0, global_fov_idx  # fallback

    def notify_zarr_frame(
        self, t: int, fov_idx: int, z: int, channel: str, region_idx: int = 0
    ):
        """Notify viewer that a new frame was written to the zarr store.

        Thread-safe: can be called from acquisition worker thread.
        Call this after each frame is written to the zarr store.

        Args:
            t: Timepoint index
            fov_idx: FOV index within region (0-based). For non-region modes, this
                     is the global FOV index.
            z: Z-level index
            channel: Channel name
            region_idx: Region index for 6d_regions mode (default 0 for backward compat)
        """
        # Map channel name to index
        channel_idx = self._zarr_channel_map.get(channel, -1)
        if channel_idx < 0:
            logger.warning(f"Unknown channel '{channel}' in notify_zarr_frame")
            return

        # Convert to global FOV index for 6d_regions mode
        global_fov_idx = fov_idx
        if self._zarr_6d_regions_mode and self._region_fov_offsets:
            if region_idx < len(self._region_fov_offsets):
                global_fov_idx = self._region_fov_offsets[region_idx] + fov_idx
            else:
                logger.warning(
                    f"Invalid region_idx {region_idx} (max: {len(self._region_fov_offsets) - 1})"
                )

        # Track this plane as written (for cache eligibility)
        cache_key = ("zarr", t, global_fov_idx, z, channel_idx)
        with self._zarr_written_planes_lock:
            self._zarr_written_planes.add(cache_key)

        # Invalidate any cached entry for this plane. This handles the race condition
        # where the viewer read zeros before the data was flushed to disk.
        self._plane_cache.invalidate(cache_key)

        # Emit signal for main thread handling
        try:
            self._zarr_frame_registered.emit(t, global_fov_idx, z, channel_idx)
        except RuntimeError as e:
            logger.warning(
                "Could not emit zarr_frame_registered signal (viewer may be closed): %s",
                e,
            )

    def _on_zarr_frame_registered(self, t: int, fov_idx: int, z: int, channel_idx: int):
        """Handle zarr frame registration signal (runs on main thread).

        Updates slider ranges and schedules debounced load if needed.
        """
        try:
            # Update per-timepoint max FOV tracking
            current_max_for_t = self._max_fov_per_time.get(t, -1)
            if fov_idx > current_max_for_t:
                self._max_fov_per_time[t] = fov_idx

            # Compute max time
            new_max_t = max(self._max_time_idx, t)

            self._updating_sliders = True
            try:
                # Update T slider if needed
                if new_max_t > self._max_time_idx:
                    self._max_time_idx = new_max_t
                    self._time_slider.setMaximum(new_max_t)

                # Show T slider if we have multiple timepoints
                if new_max_t > 0:
                    self._time_container.setVisible(True)

                # Update FOV slider max for CURRENT timepoint only
                if t == self._current_time_idx:
                    current_fov_max = self._fov_slider.maximum()
                    available_fov_max = self._max_fov_per_time.get(t, 0)
                    if available_fov_max > current_fov_max:
                        self._fov_slider.setMaximum(available_fov_max)
            finally:
                self._updating_sliders = False

            # Schedule debounced load if this frame is for the current FOV
            if t == self._current_time_idx and fov_idx == self._current_fov_idx:
                self._schedule_zarr_debounced_load()
        except Exception as e:
            logger.error("Error in _on_zarr_frame_registered: %s", e, exc_info=True)

    def _schedule_zarr_debounced_load(self):
        """Schedule a debounced load from the zarr store.

        Coalesces rapid frame registrations into a single load every 200ms.
        """
        self._zarr_load_pending = True

        if self._zarr_debounce_timer is None:
            self._zarr_debounce_timer = QTimer(self)
            self._zarr_debounce_timer.setSingleShot(True)
            self._zarr_debounce_timer.timeout.connect(self._execute_zarr_debounced_load)

        if not self._zarr_debounce_timer.isActive():
            self._zarr_debounce_timer.start(ZARR_LOAD_DEBOUNCE_MS)

    def _execute_zarr_debounced_load(self):
        """Execute the debounced zarr load."""
        if self._zarr_load_pending:
            self._zarr_load_pending = False
            self._load_current_zarr_fov()

    def _load_zarr_plane(
        self, t: int, fov_idx: int, z: int, channel_idx: int
    ) -> np.ndarray:
        """Load a single plane from the zarr store using tensorstore.

        Uses tensorstore to support both zarr v2 and v3 formats.
        Supports two modes:
        - 6D regions mode: each region has a 6D array (FOV, T, C, Z, Y, X)
        - Per-FOV mode: each FOV has a 5D array (T, C, Z, Y, X)

        Args:
            t: Timepoint index
            fov_idx: FOV index (global index, converted to region/local for 6D mode)
            z: Z-level index
            channel_idx: Channel index

        Returns:
            Image plane as numpy array, or zeros if data not available or
            no zarr mode is configured.
        """
        cache_key = ("zarr", t, fov_idx, z, channel_idx)

        # Check cache first
        cached_plane = self._plane_cache.get(cache_key)
        if cached_plane is not None:
            return cached_plane

        # Record if plane was marked as written BEFORE we read. This prevents a race
        # where notify_zarr_frame is called during our read - we shouldn't cache
        # potentially stale data that was read before the notification.
        with self._zarr_written_planes_lock:
            was_written_before_read = cache_key in self._zarr_written_planes

        # Determine which store to use based on mode
        arr = None

        if self._zarr_6d_regions_mode:
            # 6D regions mode: each region has its own store with (FOV, T, C, Z, Y, X)
            region_idx, local_fov_idx = self._global_to_region_fov(fov_idx)

            if region_idx >= len(self._zarr_region_paths):
                logger.warning(
                    f"Region index {region_idx} out of range "
                    f"(max: {len(self._zarr_region_paths) - 1})"
                )
                return np.zeros(
                    (self._image_height, self._image_width), dtype=np.uint16
                )

            # Get or open the store for this region (lazy loading, thread-safe)
            with self._zarr_stores_lock:
                if region_idx not in self._zarr_region_stores:
                    region_path = self._zarr_region_paths[region_idx]
                    logger.debug(
                        f"Opening zarr store for region {region_idx}: {region_path}"
                    )
                    # 6D mode: array is directly at acquisition.zarr, not at /0
                    ts_arr = open_zarr_tensorstore(region_path, array_path="")
                    if ts_arr is None:
                        logger.debug(
                            f"Zarr store not accessible for region {region_idx}"
                        )
                        return np.zeros(
                            (self._image_height, self._image_width), dtype=np.uint16
                        )
                    self._zarr_region_stores[region_idx] = ts_arr
                arr = self._zarr_region_stores[region_idx]

            # Read from 6D array: (FOV, T, C, Z, Y, X)
            try:
                # Index 6D: (FOV, T, C, Z, Y, X) → arr[fov, t, c, z, :, :]
                plane = arr[local_fov_idx, t, channel_idx, z, :, :].read().result()
                plane = np.asarray(plane)
                # Cache if not in live acquisition, or if plane was written before we read.
                # Using was_written_before_read prevents race with notify_zarr_frame.
                if not self._zarr_acquisition_active or was_written_before_read:
                    self._plane_cache.put(cache_key, plane)
                return plane
            except (IndexError, KeyError) as e:
                logger.debug(
                    f"Zarr plane not available (region={region_idx}, fov={local_fov_idx}, "
                    f"t={t}, z={z}, ch={channel_idx}): {e}"
                )
                return np.zeros(
                    (self._image_height, self._image_width), dtype=np.uint16
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load zarr plane (region={region_idx}, fov={local_fov_idx}, "
                    f"t={t}, z={z}, ch={channel_idx}): {type(e).__name__}: {e}"
                )
                return np.zeros(
                    (self._image_height, self._image_width), dtype=np.uint16
                )

        elif self._zarr_fov_paths:
            # Per-FOV mode: each FOV has its own store
            if fov_idx >= len(self._zarr_fov_paths):
                logger.warning(
                    f"FOV index {fov_idx} out of range (max: {len(self._zarr_fov_paths) - 1})"
                )
                return np.zeros(
                    (self._image_height, self._image_width), dtype=np.uint16
                )

            # Get or open the store for this FOV (thread-safe)
            with self._zarr_stores_lock:
                if fov_idx not in self._zarr_fov_stores:
                    fov_path = self._zarr_fov_paths[fov_idx]
                    ts_arr = open_zarr_tensorstore(fov_path, array_path="0")
                    if ts_arr is None:
                        logger.debug(f"Zarr store not accessible for FOV {fov_idx}")
                        return np.zeros(
                            (self._image_height, self._image_width), dtype=np.uint16
                        )
                    self._zarr_fov_stores[fov_idx] = ts_arr
                arr = self._zarr_fov_stores[fov_idx]

        else:
            # No valid zarr mode configured
            logger.warning("No zarr paths configured for loading")
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

        # Per-FOV mode: read from 5D array (T, C, Z, Y, X)
        try:
            plane = arr[t, channel_idx, z, :, :].read().result()
            plane = np.asarray(plane)
            # Cache if not in live acquisition, or if plane was written before we read.
            # Using was_written_before_read prevents race with notify_zarr_frame.
            if not self._zarr_acquisition_active or was_written_before_read:
                self._plane_cache.put(cache_key, plane)
            return plane

        except (IndexError, KeyError) as e:
            # Expected errors when data not yet written
            logger.debug(
                f"Zarr plane not available (t={t}, fov={fov_idx}, z={z}, ch={channel_idx}): {e}"
            )
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)
        except Exception as e:
            # Unexpected errors - log with more visibility
            logger.warning(
                f"Failed to load zarr plane (t={t}, fov={fov_idx}, z={z}, ch={channel_idx}): "
                f"{type(e).__name__}: {e}"
            )
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

    def _load_current_zarr_fov(self):
        """Load and display data for the current FOV from zarr store.

        Creates a lazy dask array. Retries if store not ready yet.
        """
        if not self._channel_names or not self._z_levels:
            return
        if self._image_height == 0 or self._image_width == 0:
            return

        t = self._current_time_idx
        fov_idx = self._current_fov_idx
        h, w = self._image_height, self._image_width

        # Check if store is ready (zarr.json exists) before creating dask array
        # This handles per_fov mode where each FOV has its own store
        if self._zarr_fov_paths and fov_idx < len(self._zarr_fov_paths):
            zarr_json = self._zarr_fov_paths[fov_idx] / "zarr.json"
            if not zarr_json.exists():
                # Store not ready yet, retry later
                if self._zarr_acquisition_active:
                    QTimer.singleShot(500, self._load_current_zarr_fov)
                return

        # Check if store is ready for 6D regions mode
        if self._zarr_6d_regions_mode and self._zarr_region_paths:
            region_idx, _ = self._global_to_region_fov(fov_idx)
            if region_idx < len(self._zarr_region_paths):
                zarr_json = self._zarr_region_paths[region_idx] / "zarr.json"
                if not zarr_json.exists():
                    # Store not ready yet, retry later
                    if self._zarr_acquisition_active:
                        QTimer.singleShot(500, self._load_current_zarr_fov)
                    return

        import dask
        import dask.array as da

        # Create lazy dask array
        delayed_planes = []
        for z in self._z_levels:
            channel_planes = []
            for c_idx in range(len(self._channel_names)):
                delayed_load = dask.delayed(self._load_zarr_plane)(t, fov_idx, z, c_idx)
                da_plane = da.from_delayed(delayed_load, shape=(h, w), dtype=np.uint16)
                channel_planes.append(da_plane)
            delayed_planes.append(da.stack(channel_planes))
        data = da.stack(delayed_planes)

        # Update NDV viewer data without rebuilding
        self._update_ndv_data(data)

    def end_zarr_acquisition(self):
        """Mark zarr acquisition as ended.

        Call this when acquisition completes. FOV labels are preserved
        for post-acquisition navigation.
        """
        if self._zarr_debounce_timer and self._zarr_debounce_timer.isActive():
            self._zarr_debounce_timer.stop()
        self._zarr_load_pending = False

        self._zarr_acquisition_active = False
        # Keep zarr store open for browsing
        logger.info("NDViewer: Zarr acquisition ended")

    def is_zarr_push_mode_active(self) -> bool:
        """Check if zarr push-based mode is active."""
        return (
            self._zarr_acquisition_active
            or bool(self._zarr_fov_paths)
            or self._zarr_6d_regions_mode
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy live refresh (kept for existing dataset viewing)
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_live_refresh(self):
        """Poll the dataset folder periodically to pick up new timepoints during acquisition."""
        # Only enable when lazy loading + NDV are available; otherwise refresh does nothing useful.
        if not (LAZY_LOADING_AVAILABLE and NDV_AVAILABLE and self.ndv_viewer):
            return
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(LIVE_REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._maybe_refresh)
        self._refresh_timer.start()

    def _close_tiff_handles(self, handles):
        """Close a list of TiffFile handles, logging any errors."""
        for h in handles or []:
            try:
                h.close()
            except Exception as e:
                logger.debug("Failed to close TiffFile handle: %s", e)

    def _close_open_handles(self):
        """Close mmap TiffFile handles (OME path) from the previously loaded dataset."""
        self._close_tiff_handles(getattr(self, "_open_handles", []))
        self._open_handles = []

    def _close_tiff_handle_cache(self):
        """Close cached TiffFile handles used by push-based OME-TIFF loading."""
        with self._tiff_handles_lock:
            entries = list(self._tiff_handles.values())
            self._tiff_handles.clear()
        # Acquire each per-file lock before closing to wait for in-flight readers.
        for tif, file_lock in entries:
            with file_lock:
                self._close_tiff_handles([tif])

    def closeEvent(self, event):
        """Clean up resources when the widget is closed."""
        if self._refresh_timer:
            self._refresh_timer.stop()
        if self._time_play_timer:
            self._time_play_timer.stop()
        if self._fov_play_timer:
            self._fov_play_timer.stop()
        if self._load_debounce_timer:
            self._load_debounce_timer.stop()
        if self._zarr_debounce_timer:
            self._zarr_debounce_timer.stop()
        # Clean up zarr state
        self._zarr_acquisition_active = False
        with self._zarr_stores_lock:
            self._zarr_fov_stores.clear()
            self._zarr_region_stores.clear()
        self._zarr_fov_paths = []
        # Clean up 6d_regions state
        self._zarr_region_paths = []
        self._fovs_per_region = []
        self._region_fov_offsets = []
        self._zarr_6d_regions_mode = False
        with self._zarr_written_planes_lock:
            self._zarr_written_planes.clear()
        self._close_open_handles()
        self._close_tiff_handle_cache()
        super().closeEvent(event)

    def _force_refresh(self):
        self._last_sig = None
        self._maybe_refresh()

    def _dataset_signature(self) -> tuple:
        """Return a cheap signature that changes when new data likely arrived."""
        base = Path(self.dataset_path)
        fmt = detect_format(base)

        if fmt == "zarr_v3":
            # For zarr v3, check zarr.json mtime and acquisition_complete flag
            fovs, structure_type = discover_zarr_v3_fovs(base)
            if not fovs:
                return (fmt, 0, False, 0)

            # Get first zarr path to check metadata
            first_path = fovs[0]["path"]
            zarr_json_path = first_path / "zarr.json"

            mtime_ns = 0
            acquisition_complete = False

            if zarr_json_path.exists():
                try:
                    st = zarr_json_path.stat()
                    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                    meta = parse_zarr_v3_metadata(first_path)
                    acquisition_complete = meta.get("acquisition_complete", False)
                except Exception as e:
                    logger.debug("Error reading zarr metadata: %s", e)

            return (fmt, len(fovs), acquisition_complete, mtime_ns)

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
        """Update ndv data in-place to avoid memory leak (ndv#209).

        Bypasses ndv's data setter which leaks GPU handles. When the data
        shape changes, emits dims_changed to trigger slider updates without
        rebuilding the entire viewer.

        Args:
            data: The new xarray DataArray to display.

        Returns:
            True if in-place update succeeded, False if caller should
            fall back to _set_ndv_data() for a full viewer rebuild.

        Note:
            Relies on ndv internal APIs (_data_model.data_wrapper._data).
            Tested with ndv 0.4.0. May need updating if ndv internals change.
        """
        v = self.ndv_viewer
        if v is None:
            return False

        try:
            wrapper = v._data_model.data_wrapper
            if wrapper._data is None:
                return False

            shape_changed = wrapper._data.shape != data.shape
            wrapper._data = data

            if shape_changed:
                # Emit dims_changed signal to update slider ranges without full rebuild.
                # In ndv, this signal triggers _fully_synchronize_view() which recreates
                # sliders based on the new data shape.
                wrapper.dims_changed.emit()
            else:
                v._request_data()

            return True
        except AttributeError as e:
            # Expected when ndv version doesn't have the expected internal structure
            logger.debug("In-place update unavailable (ndv API mismatch): %s", e)
            return False
        except Exception as e:
            # Unexpected error - log for debugging but allow fallback
            logger.warning(
                "In-place ndv update failed unexpectedly: %s", e, exc_info=True
            )
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

        data = self._create_lazy_array(Path(self.dataset_path))
        if data is None:
            return

        # Update signature only after we've confirmed we'll swap data
        self._last_sig = sig

        # Swap dataset, keeping OME handles alive for the new data
        old_data = self._xarray_data
        old_handles = getattr(self, "_open_handles", [])
        self._xarray_data = data
        self._open_handles = data.attrs.get("_open_tifs", [])

        # Check if data structure changed (dims, channels, channel names, or LUTs) - if so, force full rebuild
        structure_changed = self._data_structure_changed(old_data, data)

        # Prefer in-place update to avoid visible refresh, but only if structure unchanged.
        # When structure changes (e.g., different channels), we must rebuild the viewer
        # to avoid stale channel controls persisting from the previous dataset.
        if not structure_changed and self._try_inplace_ndv_update(data):
            # Update channel labels for the new data
            self._initiate_channel_label_update()
            # Close old handles after successful swap.
            self._close_tiff_handles(old_handles)
            return

        # Fallback: rebuild widget (may be visible on some platforms). Reduce flicker a bit.
        reason = (
            "Data structure changed" if structure_changed else "In-place update failed"
        )
        logger.debug("%s, performing full viewer rebuild", reason)

        try:
            self.setUpdatesEnabled(False)
            self._set_ndv_data(data)
        finally:
            self.setUpdatesEnabled(True)
            # Close old handles regardless.
            self._close_tiff_handles(old_handles)

    def _data_structure_changed(
        self, old_data: Optional["xr.DataArray"], new_data: "xr.DataArray"
    ) -> bool:
        """Check if data structure changed significantly (requiring full viewer rebuild).

        Delegates to the module-level :func:`data_structure_changed` function,
        wrapping it with exception handling for safety in the viewer context.

        Args:
            old_data: Previous dataset state (or None for first load).
            new_data: Newly loaded dataset to compare.

        Returns:
            True if structure changed and viewer needs full rebuild.
        """
        try:
            return data_structure_changed(old_data, new_data)
        except Exception as e:
            # On any error, assume structure changed to be safe
            logger.debug("Error checking data structure change: %s", e)
            return True

    def load_dataset(self, path: str):
        """Load dataset and display in NDV."""
        # Close any previously open file handles before loading new dataset
        self._close_open_handles()

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

    def set_current_index(self, dim: str, value: int) -> bool:
        """Set the current index for a dimension in the viewer.

        Programmatically navigate the viewer to a specific position along
        a dimension (e.g., 'fov', 'time', 'z', 'channel').

        Args:
            dim: Dimension name (must exist in the loaded data).
            value: Index value to set.

        Returns:
            True if successful, False otherwise.
        """
        if self.ndv_viewer is None:
            logger.debug("set_current_index: no viewer available")
            return False

        try:
            # NDV ArrayViewer uses display_model.current_index
            if hasattr(self.ndv_viewer, "display_model"):
                dm = self.ndv_viewer.display_model
                if hasattr(dm, "current_index") and dim in dm.current_index:
                    dm.current_index[dim] = value
                    logger.debug(f"set_current_index: {dim}={value}")
                    return True

            # Fallback for older NDV versions using dims API
            if hasattr(self.ndv_viewer, "dims"):
                dims = self.ndv_viewer.dims
                if hasattr(dims, "current_step"):
                    current = dict(dims.current_step)
                    if dim in current:
                        current[dim] = value
                        dims.current_step = current
                        logger.debug(f"set_current_index (fallback): {dim}={value}")
                        return True

            logger.debug(
                f"set_current_index: dimension '{dim}' not found or API unavailable"
            )
            return False
        except Exception as e:
            logger.debug(f"set_current_index error: {e}")
            return False

    def get_fov_list(self) -> List[Dict]:
        """Get the list of FOVs for the currently loaded dataset.

        Returns a list of dicts with 'region' (well ID) and 'fov' (FOV index)
        keys, sorted by region then FOV. This can be used to map between
        (well_id, fov_index) pairs and flat xarray FOV dimension indices.

        Returns:
            List of {"region": str, "fov": int} dicts, or empty list if no
            dataset is loaded.
        """
        if not getattr(self, "dataset_path", ""):
            return []

        try:
            base_path = Path(self.dataset_path)
            fmt = detect_format(base_path)
            return self._discover_fovs(base_path, fmt)
        except Exception as e:
            logger.debug(f"get_fov_list error: {e}")
            return []

    def has_fov_dimension(self) -> bool:
        """Check if loaded data has an FOV dimension.

        Returns:
            True if data is loaded and has 'fov' dimension, False otherwise.
        """
        xarray_data = getattr(self, "_xarray_data", None)
        if xarray_data is None:
            return False
        return "fov" in xarray_data.dims

    def refresh(self) -> None:
        """Force an immediate refresh of the viewer display.

        Useful after loading a new dataset or when you want to update
        the display without waiting for the automatic refresh timer.
        """
        self._force_refresh()

    def _create_lazy_array(self, base_path: Path) -> "Optional[xr.DataArray]":
        """Create lazy xarray from dataset - auto-detects format."""
        if not LAZY_LOADING_AVAILABLE:
            return None

        fmt = detect_format(base_path)

        # Zarr v3 has its own FOV discovery
        if fmt == "zarr_v3":
            fovs, structure_type = discover_zarr_v3_fovs(base_path)
            if not fovs:
                print("No zarr v3 FOVs found")
                return None
            return self._load_zarr_v3(base_path, fovs, structure_type)

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
    ) -> "Optional[xr.DataArray]":
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

            # Squeeze out singleton fov/time to avoid showing "0 of 0" sliders
            if n_fov == 1:
                xarr = xarr.isel(fov=0, drop=True)
            if n_t == 1:
                xarr = xarr.isel(time=0, drop=True)

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
    ) -> "Optional[xr.DataArray]":
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

            # Build human-readable FOV labels (e.g. "A1:0", "A1:1", "B2:0")
            # so the ndv slider shows the well:index name instead of a bare
            # integer. fovs is a list of {"region", "fov"} dicts.
            fov_coords = [f"{f['region']}:{f['fov']}" for f in fovs]
            xarr = xr.DataArray(
                stacked,
                dims=["time", "fov", "z", "channel", "y", "x"],
                # Use actual values for time/z coords; FOV uses readable
                # well:index labels (when meaningful regions exist) so the
                # ndv slider displays "A1:0" rather than a bare index.
                coords={
                    "time": times,
                    "fov": fov_coords,
                    "z": z_levels,
                    "channel": list(range(n_c)),
                },
            )

            # Squeeze out singleton dimensions to avoid showing "0 of 0" sliders
            if n_fov == 1:
                xarr = xarr.isel(fov=0, drop=True)
            if n_t == 1:
                xarr = xarr.isel(time=0, drop=True)

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

    def _load_zarr_v3(
        self, base_path: Path, fovs: List[Dict], structure_type: str
    ) -> "Optional[xr.DataArray]":
        """Load zarr v3 dataset (OME-NGFF format from Squid).

        Uses tensorstore to support both zarr v2 and v3 formats.

        Args:
            base_path: Dataset root directory
            fovs: List of FOV dicts from discover_zarr_v3_fovs
            structure_type: "hcs_plate", "per_fov", or "6d"

        Returns:
            xarray.DataArray with dims (time, fov, z, channel, y, x)
        """
        try:
            if structure_type == "6d":
                return self._load_zarr_v3_6d(fovs[0]["path"])
            else:
                return self._load_zarr_v3_5d(fovs)
        except Exception as e:
            print(f"Zarr v3 load error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _load_zarr_v3_5d(self, fovs: List[Dict]) -> "Optional[xr.DataArray]":
        """Load per-FOV zarr v3 stores and stack them.

        Each FOV is a separate zarr store with dimensions (T, C, Z, Y, X).
        Stacks them along a new FOV axis.

        Uses tensorstore to support both zarr v2 and v3 formats.
        """
        import dask.array as da
        import xarray as xr

        if not fovs:
            return None

        # Parse metadata from first FOV
        first_path = fovs[0]["path"]
        meta = parse_zarr_v3_metadata(first_path)

        # Open first zarr to get shape/dtype using tensorstore
        ts_arr = open_zarr_tensorstore(first_path, array_path="0")
        if ts_arr is None:
            logger.error("Failed to open zarr store %s", first_path)
            return None

        shape = ts_arr.shape
        dtype = ts_arr.dtype.numpy_dtype

        # Determine axis order from metadata or infer from shape
        # Typical OME-NGFF order: (T, C, Z, Y, X)
        axes = meta.get("axes", [])
        if axes:
            axis_names = [ax.get("name", "").lower() for ax in axes]
        else:
            # Infer from shape length
            if len(shape) == 5:
                axis_names = ["t", "c", "z", "y", "x"]
            elif len(shape) == 4:
                axis_names = ["c", "z", "y", "x"]
            elif len(shape) == 3:
                axis_names = ["z", "y", "x"]
            else:
                axis_names = ["y", "x"]

        # Extract dimensions
        n_t = shape[axis_names.index("t")] if "t" in axis_names else 1
        n_c = shape[axis_names.index("c")] if "c" in axis_names else 1
        n_z = shape[axis_names.index("z")] if "z" in axis_names else 1
        n_y = shape[axis_names.index("y")] if "y" in axis_names else shape[-2]
        n_x = shape[axis_names.index("x")] if "x" in axis_names else shape[-1]

        # Build channel info
        channel_names = meta.get("channel_names", [])
        if not channel_names or len(channel_names) != n_c:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        channel_colors = meta.get("channel_colors", [])
        luts = {}
        for i, name in enumerate(channel_names):
            if i < len(channel_colors) and channel_colors[i]:
                luts[i] = hex_to_colormap(channel_colors[i])
            else:
                luts[i] = wavelength_to_colormap(extract_wavelength(name))

        # Create dask arrays for each FOV using tensorstore
        fov_arrays = []
        for fov_info in fovs:
            fov_path = fov_info["path"]
            try:
                ts_arr = open_zarr_tensorstore(fov_path, array_path="0")
                if ts_arr is None:
                    raise RuntimeError(f"Could not open zarr store at {fov_path}")

                # Create dask array with per-plane chunks.
                # Wrap tensorstore array so dask sees a numpy-compatible dtype.
                chunks = tuple(
                    1 if i < len(shape) - 2 else s for i, s in enumerate(shape)
                )
                darr = da.from_array(_TensorStoreArrayWrapper(ts_arr), chunks=chunks)

                # Ensure shape is (T, C, Z, Y, X)
                if len(darr.shape) == 4:
                    darr = darr[np.newaxis, ...]  # Add T axis
                elif len(darr.shape) == 3:
                    darr = darr[np.newaxis, np.newaxis, ...]  # Add T and C axes

                fov_arrays.append(darr)
            except Exception as e:
                logger.warning("Failed to load FOV %s: %s", fov_path, e)
                # Create zeros placeholder
                fov_arrays.append(
                    da.zeros(
                        (n_t, n_c, n_z, n_y, n_x),
                        dtype=dtype,
                        chunks=(1, 1, 1, n_y, n_x),
                    )
                )

        # Stack FOVs: (n_fov, T, C, Z, Y, X)
        stacked = da.stack(fov_arrays, axis=0)

        # Transpose to (T, FOV, Z, C, Y, X) then reorder to standard dims
        # Current: (FOV, T, C, Z, Y, X)
        # Want: (T, FOV, Z, C, Y, X)
        stacked = da.moveaxis(stacked, 0, 1)  # Now (T, FOV, C, Z, Y, X)
        stacked = da.moveaxis(stacked, 3, 2)  # Now (T, FOV, Z, C, Y, X)

        n_fov = len(fovs)
        # Build readable FOV labels so the ndv slider shows the well:index
        # name (e.g. "A1:0") rather than a bare integer.
        fov_coords = [f"{f['region']}:{f['fov']}" for f in fovs]
        xarr = xr.DataArray(
            stacked,
            dims=["time", "fov", "z", "channel", "y", "x"],
            coords={
                "time": list(range(n_t)),
                "fov": fov_coords,
                "z": list(range(n_z)),
                "channel": list(range(n_c)),
            },
        )
        xarr.attrs["luts"] = luts
        xarr.attrs["channel_names"] = channel_names

        # Store physical sizes
        if meta.get("pixel_size_um") is not None:
            xarr.attrs["pixel_size_um"] = meta["pixel_size_um"]
            xarr.attrs["pixel_size_x_um"] = meta["pixel_size_um"]
            xarr.attrs["pixel_size_y_um"] = meta["pixel_size_um"]
        if meta.get("dz_um") is not None:
            xarr.attrs["dz_um"] = meta["dz_um"]
            xarr.attrs["pixel_size_z_um"] = meta["dz_um"]

        return xarr

    def _load_zarr_v3_6d(self, zarr_path: Path) -> "Optional[xr.DataArray]":
        """Load a single 6D zarr v3 store with FOV dimension.

        Handles zarr stores with dimensions like (T, FOV, C, Z, Y, X).
        Uses tensorstore to support both zarr v2 and v3 formats.
        """
        import dask.array as da
        import xarray as xr

        meta = parse_zarr_v3_metadata(zarr_path)

        # Open zarr store using tensorstore
        ts_arr = open_zarr_tensorstore(zarr_path, array_path="0")
        if ts_arr is None:
            logger.error("Failed to open zarr store %s", zarr_path)
            return None
        shape = ts_arr.shape

        # Determine axis order
        axes = meta.get("axes", [])
        if axes:
            axis_names = [ax.get("name", "").lower() for ax in axes]
        else:
            # Fallback: infer axis order from shape length when metadata is missing.
            # Assumes OME-NGFF conventions (T, C, Z, Y, X) or (T, FOV, C, Z, Y, X).
            if len(shape) == 6:
                axis_names = ["t", "fov", "c", "z", "y", "x"]
            elif len(shape) == 5:
                axis_names = ["t", "c", "z", "y", "x"]
            else:
                axis_names = (
                    ["c", "z", "y", "x"] if len(shape) == 4 else ["z", "y", "x"]
                )

        # Normalize axis names
        axis_map = {"position": "fov", "p": "fov", "time": "t"}
        axis_names = [axis_map.get(n, n) for n in axis_names]

        # Extract dimensions
        def get_dim(name, default=1):
            if name in axis_names:
                return shape[axis_names.index(name)]
            return default

        n_t = get_dim("t")
        n_fov = get_dim("fov")
        n_c = get_dim("c")
        n_z = get_dim("z")
        # n_y and n_x not needed - shape comes from dask array directly

        # Build channel info
        channel_names = meta.get("channel_names", [])
        if not channel_names or len(channel_names) != n_c:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        channel_colors = meta.get("channel_colors", [])
        luts = {}
        for i, name in enumerate(channel_names):
            if i < len(channel_colors) and channel_colors[i]:
                luts[i] = hex_to_colormap(channel_colors[i])
            else:
                luts[i] = wavelength_to_colormap(extract_wavelength(name))

        # Create dask array with per-plane chunks using tensorstore array.
        # Wrap tensorstore array so dask sees a numpy-compatible dtype.
        chunks = tuple(1 if i < len(shape) - 2 else s for i, s in enumerate(shape))
        darr = da.from_array(_TensorStoreArrayWrapper(ts_arr), chunks=chunks)

        # Transpose to standard order: (time, fov, z, channel, y, x)
        # Build transpose order based on current axis order
        target_order = ["t", "fov", "z", "c", "y", "x"]
        current_order = list(axis_names)  # Copy to avoid mutating original

        # Ensure all target axes exist
        for ax in target_order:
            if ax not in current_order:
                # Add missing dimension
                darr = darr[..., np.newaxis]
                current_order.append(ax)

        # Build transpose indices
        transpose_idx = []
        for ax in target_order:
            if ax in current_order:
                transpose_idx.append(current_order.index(ax))

        if transpose_idx != list(range(len(transpose_idx))):
            darr = da.transpose(darr, transpose_idx)

        # Reshape if needed to ensure 6D
        while len(darr.shape) < 6:
            darr = darr[np.newaxis, ...]

        xarr = xr.DataArray(
            darr,
            dims=["time", "fov", "z", "channel", "y", "x"],
            coords={
                "time": list(range(n_t)),
                "fov": list(range(n_fov)),
                "z": list(range(n_z)),
                "channel": list(range(n_c)),
            },
        )
        xarr.attrs["luts"] = luts
        xarr.attrs["channel_names"] = channel_names

        # Store physical sizes
        if meta.get("pixel_size_um") is not None:
            xarr.attrs["pixel_size_um"] = meta["pixel_size_um"]
            xarr.attrs["pixel_size_x_um"] = meta["pixel_size_um"]
            xarr.attrs["pixel_size_y_um"] = meta["pixel_size_um"]
        if meta.get("dz_um") is not None:
            xarr.attrs["dz_um"] = meta["dz_um"]
            xarr.attrs["pixel_size_z_um"] = meta["dz_um"]

        return xarr

    def _set_ndv_data(self, data: "xr.DataArray"):
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

        # Update scale bar overlay and add tooltips to ndv controls (deferred)
        self._setup_scale_bar(data)
        self._setup_fov_overlay(data)
        QTimer.singleShot(500, self._add_ndv_tooltips)
        QTimer.singleShot(600, self._hook_dim_toggle)

        # Update channel labels after viewer is ready.
        self._initiate_channel_label_update()

    def _add_ndv_tooltips(self):
        """Add tooltips to ndv's internal controls."""
        try:
            qwidget = self.ndv_viewer.widget()
            for child in qwidget.findChildren(QWidget):
                cls_name = type(child).__name__
                if cls_name == "ROIButton":
                    child.setToolTip("Draw a rectangular region of interest on the image")
                elif cls_name == "_DimToggleButton":
                    child.setToolTip("Toggle between 2D and 3D view")
                elif cls_name == "QComboBox" and hasattr(child, "currentText"):
                    text = child.currentText().lower()
                    if text in ("composite", "grayscale", "rgba"):
                        child.setToolTip("Channel display mode: composite, grayscale, or RGBA")
            for btn in qwidget.findChildren(QPushButton):
                if not btn.text() and btn.toolTip() == "":
                    btn.setToolTip("Reset zoom to fit image")
                    break
            logger.info("Added tooltips to ndv controls")
        except Exception as e:
            logger.debug("Could not add ndv tooltips: %s", e)

    def _hook_dim_toggle(self):
        """Connect to ndv's 2D/3D toggle button to hide scale bar in 3D mode."""
        try:
            qwidget = self.ndv_viewer.widget()
            for child in qwidget.findChildren(QWidget):
                if type(child).__name__ == "_DimToggleButton":
                    child.clicked.connect(self._on_dim_toggled)
                    break
        except Exception as e:
            logger.debug("Could not hook dim toggle: %s", e)

    def _on_dim_toggled(self):
        """Called when the 2D/3D toggle button is clicked."""
        if not hasattr(self, "_scale_bar_widget"):
            return
        # Defer so ndv has time to switch camera
        QTimer.singleShot(400, self._reconnect_after_dim_toggle)

    def _reconnect_after_dim_toggle(self):
        """Reconnect camera events after 2D/3D toggle and redraw scale bar."""
        self._connect_camera_events()
        QTimer.singleShot(100, self._redraw_scale_bar)

    def _setup_scale_bar(self, data):
        """Set up a scale bar that updates with camera zoom."""
        self._pixel_size_um = data.attrs.get("pixel_size_um")
        if self._pixel_size_um is None:
            if hasattr(self, "_scale_bar_widget"):
                self._scale_bar_widget.setVisible(False)
            return

        # Find the vispy canvas widget inside ndv
        canvas_widget = None
        try:
            qwidget = self.ndv_viewer.widget()
            if hasattr(qwidget, '_canvas_widget'):
                canvas_widget = qwidget._canvas_widget
        except Exception:
            pass
        if canvas_widget is None:
            canvas_widget = self.ndv_viewer.widget()

        # Create or reuse the scale bar widget, parented to the canvas
        if not hasattr(self, "_scale_bar_widget"):
            self._scale_bar_widget = _ScaleBarWidget(canvas_widget)
        else:
            self._scale_bar_widget.setParent(canvas_widget)

        self._scale_bar_canvas = canvas_widget
        self._scale_bar_widget.setVisible(True)
        self._scale_bar_widget.raise_()

        canvas_widget.installEventFilter(self)

        # Defer camera hook — the ArrayViewer needs time to finalize its canvas/camera
        QTimer.singleShot(300, self._connect_camera_events)
        QTimer.singleShot(400, self._redraw_scale_bar)

    def _setup_fov_overlay(self, data):
        """Bottom-left overlay that shows the current FOV name.

        Mirrors _setup_scale_bar's strategy: parent a small widget to ndv's
        canvas so it floats over the image. The label text is the current
        coord value of the "fov" dim — readable strings like "A1:0" come
        from the loaders that build human FOV labels; numeric coords are
        shown as-is.
        """
        # Hide overlay if there is no FOV dimension or only a single FOV
        if "fov" not in data.dims or data.sizes.get("fov", 1) <= 1:
            if hasattr(self, "_fov_overlay_label"):
                self._fov_overlay_label.setVisible(False)
            return

        canvas_widget = None
        try:
            qwidget = self.ndv_viewer.widget()
            if hasattr(qwidget, "_canvas_widget"):
                canvas_widget = qwidget._canvas_widget
        except Exception:
            pass
        if canvas_widget is None:
            canvas_widget = self.ndv_viewer.widget()

        if not hasattr(self, "_fov_overlay_label"):
            self._fov_overlay_label = QLabel(canvas_widget)
            self._fov_overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._fov_overlay_label.setStyleSheet(
                "background: rgba(0, 0, 0, 140); color: white; "
                "padding: 4px 8px; border-radius: 4px; font-size: 11px;"
            )
        else:
            self._fov_overlay_label.setParent(canvas_widget)

        self._fov_overlay_canvas = canvas_widget
        self._fov_overlay_label.setVisible(True)
        self._fov_overlay_label.raise_()

        # Live updates: listen for ndv's current_index changes if available.
        try:
            dm = getattr(self.ndv_viewer, "display_model", None)
            if dm is not None and hasattr(dm, "events"):
                ev = getattr(dm.events, "current_index", None)
                if ev is not None and not getattr(self, "_fov_overlay_connected", False):
                    ev.connect(self._update_fov_overlay)
                    self._fov_overlay_connected = True
        except Exception:
            pass

        # Initial paint + position (deferred so the canvas has its real size)
        QTimer.singleShot(300, self._update_fov_overlay)
        QTimer.singleShot(400, self._position_fov_overlay)

    def _update_fov_overlay(self, *_):
        """Refresh the bottom-left overlay text from the current FOV index."""
        if not hasattr(self, "_fov_overlay_label"):
            return
        data = getattr(self, "_xarray_data", None)
        if data is None or "fov" not in getattr(data, "dims", ()):
            self._fov_overlay_label.setVisible(False)
            return
        try:
            dm = self.ndv_viewer.display_model
            idx = dm.current_index.get("fov", 0)
        except Exception:
            idx = 0
        try:
            coord = data.coords["fov"].values
            label = str(coord[int(idx)]) if 0 <= int(idx) < len(coord) else str(idx)
        except Exception:
            label = str(idx)
        self._fov_overlay_label.setText(f"FOV: {label}")
        self._fov_overlay_label.adjustSize()
        self._position_fov_overlay()

    def _position_fov_overlay(self):
        """Bottom-left placement, mirror of the scale bar's bottom-right."""
        if not hasattr(self, "_fov_overlay_label"):
            return
        canvas = getattr(self, "_fov_overlay_canvas", None)
        if canvas is None:
            return
        ch = canvas.height()
        sh = self._fov_overlay_label.height()
        self._fov_overlay_label.move(12, ch - sh - 12)
        self._fov_overlay_label.raise_()

    def _connect_camera_events(self):
        """Connect to vispy camera transform events (deferred to ensure camera exists)."""
        try:
            vispy_canvas = self.ndv_viewer._canvas
            cam = vispy_canvas._view.camera
            cam.events.transform_change.connect(self._on_camera_changed)
            # Also catch direct mouse wheel on the scene canvas
            vispy_canvas._canvas.events.mouse_wheel.connect(self._on_camera_changed)
            self._vispy_canvas = vispy_canvas
        except Exception:
            pass

    def _on_camera_changed(self, event=None):
        """Called when vispy camera transform changes (zoom/pan)."""
        self._redraw_scale_bar()

    def _redraw_scale_bar(self):
        """Recompute scale bar size based on current zoom level."""
        if not hasattr(self, "_pixel_size_um") or self._pixel_size_um is None:
            return
        if not hasattr(self, "_scale_bar_widget"):
            return
        if not hasattr(self, "_vispy_canvas"):
            return

        # Compute zoom: screen_pixels / data_pixels
        # 2D PanZoom camera has rect; 3D Turntable camera has scale_factor
        try:
            cam = self._vispy_canvas._view.camera
            canvas_w = self._scale_bar_canvas.width()
            if canvas_w <= 0:
                return
            rect = getattr(cam, 'rect', None)
            if rect is not None and rect.width > 0:
                pixels_per_data_pixel = canvas_w / rect.width
            else:
                scale_factor = getattr(cam, 'scale_factor', None)
                if scale_factor is None or scale_factor <= 0:
                    return
                pixels_per_data_pixel = canvas_w / scale_factor
        except Exception:
            return

        # How many µm fit in ~120 screen pixels at this zoom?
        target_screen_px = 120
        target_um = (target_screen_px / pixels_per_data_pixel) * self._pixel_size_um

        # Snap to a nice round number
        nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        bar_um = min(nice, key=lambda x: abs(x - target_um))

        # Convert chosen physical length back to screen pixels
        bar_data_px = bar_um / self._pixel_size_um
        bar_screen_px = int(bar_data_px * pixels_per_data_pixel)
        bar_screen_px = max(bar_screen_px, 20)
        bar_screen_px = min(bar_screen_px, canvas_w - 40)

        if bar_um >= 1000:
            text = f"{bar_um / 1000:.0f} mm"
        elif bar_um >= 1:
            text = f"{int(bar_um)} \u00b5m" if bar_um == int(bar_um) else f"{bar_um} \u00b5m"
        else:
            text = f"{bar_um * 1000:.0f} nm"

        self._scale_bar_widget.update_bar(bar_screen_px, text)

        # Position bottom-right of canvas
        cw = self._scale_bar_canvas.width()
        ch = self._scale_bar_canvas.height()
        sw = self._scale_bar_widget.width()
        sh = self._scale_bar_widget.height()
        self._scale_bar_widget.move(cw - sw - 12, ch - sh - 12)
        self._scale_bar_widget.raise_()

    def eventFilter(self, obj, event):
        """Reposition scale bar + FOV overlay when canvas is resized."""
        if obj is getattr(self, "_scale_bar_canvas", None):
            from PyQt5.QtCore import QEvent
            if event.type() == QEvent.Resize:
                QTimer.singleShot(0, self._redraw_scale_bar)
                QTimer.singleShot(0, self._position_fov_overlay)
        return super().eventFilter(obj, event)

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
        self.setWindowTitle(f"Cephla NDViewer Lightweight - {Path(dataset_path).name}")
        self.setGeometry(100, 100, 720, 540)  # 4:3 aspect, smaller
        self._set_dark_theme()
        _set_cephla_icon(self)

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
