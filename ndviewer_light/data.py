"""Qt-free data layer for ndviewer_light.

Zarr/tensorstore access, OME-NGFF metadata parsing, FOV/plate enumeration and the
plane cache live here so that headless consumers can read pixels without paying
for PyQt5, ndv, vispy, dask, xarray or scipy — all of which ``core`` imports at
module scope because its classes subclass ``QWidget``.

Every public name in this module is re-exported from ``ndviewer_light.core`` and
from ``ndviewer_light``; nothing here was rewritten, only moved.
"""

import json
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import tensorstore as ts

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


# Constants
TIFF_EXTENSIONS = {".tif", ".tiff"}
# Non-TIFF raster formats decoded via Pillow (e.g. Squid BMP acquisitions).
PIL_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}
# All single-image formats the single-file reader can ingest.
IMAGE_EXTENSIONS = TIFF_EXTENSIONS | PIL_EXTENSIONS


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


# Filename patterns (from common.py)
FPATTERN = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.(?:tiff?|bmp|png|jpe?g)",
    re.IGNORECASE,
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


def subset_fovs_per_region(fovs: List[Dict], n: int) -> List[Dict]:
    """Keep at most ``n`` FOVs per region (IMA-191 subset filter).

    "condition" is defined as region/well: the image data model only carries
    ``region`` and ``fov`` per image, so subsetting groups by ``region`` and
    keeps the ``n`` FOVs with the smallest integer ``fov`` index within each
    region.

    Sampling policy: first-n by *integer* fov index. Discovery iterates
    directories with lexical ``sorted()``, so ``fov_10`` would otherwise precede
    ``fov_2``; ranking by ``int(fov)`` keeps the sample deterministic and
    intuitive (fov 0, 1, 2, ... not 0, 1, 10).

    Contract:
      - ``n <= 0`` (subset disabled) or empty input -> return all FOVs unchanged.
      - A region with fewer than ``n`` FOVs keeps all of them (no error).
      - The original relative order of the returned FOVs is preserved.

    Args:
        fovs: discovery output; each dict has at least ``"region"`` and ``"fov"``.
        n: maximum FOVs to keep per region.

    Returns:
        Filtered list of FOV dicts (the same dict objects, in original order).
    """
    if n <= 0 or not fovs:
        return list(fovs)

    # Rank FOVs within each region by integer fov index and mark the first n.
    by_region: Dict[str, List[Dict]] = {}
    for entry in fovs:
        by_region.setdefault(entry["region"], []).append(entry)

    kept_ids = set()
    for entries in by_region.values():
        ranked = sorted(entries, key=lambda d: int(d["fov"]))
        for entry in ranked[:n]:
            kept_ids.add(id(entry))

    # Preserve original order; dicts are unhashable so identity-filter is used.
    return [entry for entry in fovs if id(entry) in kept_ids]


def parse_fov_label(label: str) -> Tuple[str, int]:
    """Split a push-mode FOV label into (region, fov_index).

    Labels are formatted ``"{region}:{fov}"`` (e.g. ``"A1:3"``). Falls back to
    ``(label, 0)`` when there is no integer suffix.
    """
    region, sep, fov = label.rpartition(":")
    if sep:
        try:
            return region, int(fov)
        except ValueError:
            pass
    return label, 0


def kept_fov_indices(fov_labels: List[str], n: int) -> List[int]:
    """Flat indices into ``fov_labels`` kept under an "n per region" subset.

    Push-mode counterpart of :func:`subset_fovs_per_region`. Regions are parsed
    from the labels; within each region the ``n`` FOVs with the smallest integer
    fov index are kept. Returns the kept flat indices in ascending order so the
    slider can map contiguous positions onto them.

    ``n <= 0`` returns every index (subset disabled / identity mapping), which
    keeps the slider behaving exactly as it does without the subset.
    """
    if n <= 0:
        return list(range(len(fov_labels)))

    by_region: Dict[str, List[Tuple[int, int]]] = {}
    for idx, label in enumerate(fov_labels):
        region, fov = parse_fov_label(label)
        by_region.setdefault(region, []).append((fov, idx))

    kept = set()
    for entries in by_region.values():
        for _fov, idx in sorted(entries)[:n]:
            kept.add(idx)
    return sorted(kept)


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
