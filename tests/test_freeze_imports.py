"""Tests that every conditional import used in core.py is available.

Each test is independent so failures pinpoint the exact missing package.
"""
import pytest


def test_import_ndv():
    import ndv  # noqa: F401


def test_import_ndv_vispy_canvas():
    from ndv.views._vispy._array_canvas import VispyArrayCanvas  # noqa: F401


def test_import_ndv_data_wrapper():
    from ndv.models._data_wrapper import XarrayWrapper  # noqa: F401


def test_import_superqt():
    import superqt  # noqa: F401


def test_import_superqt_iconify():
    from superqt.iconify import QIconifyIcon  # noqa: F401


def test_import_pyconify():
    import pyconify  # noqa: F401


def test_import_dask_array():
    import dask.array  # noqa: F401


def test_import_tifffile():
    import tifffile  # noqa: F401


def test_import_xarray():
    import xarray  # noqa: F401


def test_import_scipy_ndimage():
    from scipy.ndimage import zoom  # noqa: F401


def test_import_tensorstore():
    import tensorstore  # noqa: F401


def test_import_vispy():
    import vispy  # noqa: F401


def test_import_vispy_volume():
    from vispy.visuals.volume import VolumeVisual  # noqa: F401


def test_import_zarr():
    import zarr  # noqa: F401


def test_import_opengl():
    import OpenGL  # noqa: F401
    import OpenGL.GL  # noqa: F401


def test_import_pyqt5():
    from PyQt5.QtWidgets import QApplication  # noqa: F401
