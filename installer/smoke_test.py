"""Post-freeze smoke tests for the bundled ndviewer_light application."""
import os
import sys
import tempfile
import numpy


def _test(name, fn):
    """Run a single test, print PASS/FAIL, return success bool."""
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name} -- {e}")
        return False


def run():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    results = []

    def t_import_ndv():
        import ndv  # noqa: F401

    def t_vispy_array_canvas():
        from ndv.views._vispy._array_canvas import VispyArrayCanvas  # noqa: F401

    def t_data_wrapper():
        from ndv.models._data_wrapper import XarrayWrapper  # noqa: F401

    def t_import_superqt():
        import superqt  # noqa: F401

    def t_dask_array():
        import dask.array
        assert dask.array.zeros((10, 10)).compute().shape == (10, 10)

    def t_tifffile():
        import tifffile
        arr = numpy.zeros((10, 10), dtype=numpy.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, arr)
            data = tifffile.imread(path)
            assert data.shape == (10, 10)
        finally:
            os.unlink(path)

    def t_import_tensorstore():
        import tensorstore  # noqa: F401

    def t_import_xarray():
        import xarray  # noqa: F401

    def t_scipy_ndimage():
        from scipy.ndimage import zoom
        result = zoom(numpy.ones((10, 10)), 0.5)
        assert result.shape == (5, 5)

    def t_import_vispy():
        import vispy  # noqa: F401

    def t_import_zarr():
        import zarr  # noqa: F401

    def t_pyqt5():
        from PyQt5.QtWidgets import QApplication  # noqa: F401

    tests = [
        ("import ndv", t_import_ndv),
        ("ndv VispyArrayCanvas", t_vispy_array_canvas),
        ("ndv XarrayWrapper", t_data_wrapper),
        ("import superqt", t_import_superqt),
        ("dask.array compute", t_dask_array),
        ("tifffile read/write", t_tifffile),
        ("import tensorstore", t_import_tensorstore),
        ("import xarray", t_import_xarray),
        ("scipy.ndimage zoom", t_scipy_ndimage),
        ("import vispy", t_import_vispy),
        ("import zarr", t_import_zarr),
        ("PyQt5 QApplication", t_pyqt5),
    ]

    for name, fn in tests:
        results.append(_test(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} smoke tests passed.")
    sys.exit(0 if all(results) else 1)
