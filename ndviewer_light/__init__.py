"""ndviewer_light - Lightweight NDV-based viewer for microscopy data."""

try:
    from importlib.metadata import version

    __version__ = version("ndviewer_light")
except Exception:
    __version__ = "unknown"

# The package no longer imports ``core`` (and therefore PyQt5, ndv, vispy, dask,
# xarray, scipy) at import time. Names resolve on first attribute access via
# PEP 562, so ``import ndviewer_light.data`` — the Qt-free data layer — costs
# tensorstore and nothing else. ``from ndviewer_light import LightweightViewer``
# still works exactly as before; it just pays for Qt at that moment instead of
# at package import.
_DATA_NAMES = frozenset(
    {
        "FPATTERN",
        "FPATTERN_OME",
        "MemoryBoundedLRUCache",
        "data_structure_changed",
        "detect_format",
        "discover_zarr_v3_fovs",
        "extract_ome_physical_sizes",
        "extract_wavelength",
        "hex_to_colormap",
        "open_zarr_tensorstore",
        "parse_zarr_v3_metadata",
        "read_acquisition_parameters",
        "wavelength_to_colormap",
    }
)
_CORE_NAMES = frozenset(
    {
        "MAX_3D_TEXTURE_SIZE",
        "NDV_SLIDER_STYLE",
        "SLIDER_VALUE_FONT_SIZE_PX",
        "UI_FONT_SIZE_PX",
        "LightweightMainWindow",
        "LightweightViewer",
        "read_tiff_pixel_size",
    }
)


def __getattr__(name):
    if name in _DATA_NAMES:
        from . import data

        value = getattr(data, name)
    elif name in _CORE_NAMES:
        from . import core

        value = getattr(core, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _DATA_NAMES | _CORE_NAMES)


__all__ = [
    "__version__",
    "FPATTERN",
    "FPATTERN_OME",
    "MAX_3D_TEXTURE_SIZE",
    "NDV_SLIDER_STYLE",
    "SLIDER_VALUE_FONT_SIZE_PX",
    "UI_FONT_SIZE_PX",
    "LightweightMainWindow",
    "LightweightViewer",
    "MemoryBoundedLRUCache",
    "data_structure_changed",
    "detect_format",
    "discover_zarr_v3_fovs",
    "extract_ome_physical_sizes",
    "extract_wavelength",
    "hex_to_colormap",
    "open_zarr_tensorstore",
    "parse_zarr_v3_metadata",
    "read_acquisition_parameters",
    "read_tiff_pixel_size",
    "wavelength_to_colormap",
]
