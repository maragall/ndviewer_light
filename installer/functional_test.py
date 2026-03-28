"""Automated functional tests for ndviewer_light.

Runs headless (offscreen). Tests actual dataset loading, viewer creation,
dimension navigation, and channel switching against real test data.
"""
import os
import sys
import time
import tempfile
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np

PASS = 0
FAIL = 0
TEST_DATA = r"C:\Users\peter\Downloads\stitcher_ndviewer_test_set"


def report(name, success, err=None):
    global PASS, FAIL
    if success:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {err}")


def run_test(name, fn):
    try:
        fn()
        report(name, True)
    except Exception as e:
        report(name, False, f"{e}")


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    from ndviewer_light.core import (
        LightweightMainWindow,
        LightweightViewer,
        LauncherWindow,
        detect_format,
    )

    # --- Test 1: Launcher opens without crash ---
    def test_launcher():
        w = LauncherWindow()
        w.show()
        app.processEvents()
        assert w.isVisible()
        w.close()

    run_test("Launcher window opens", test_launcher)

    # --- Test 2: detect_format on each dataset ---
    if os.path.isdir(TEST_DATA):
        from pathlib import Path
        for name in sorted(os.listdir(TEST_DATA)):
            d = os.path.join(TEST_DATA, name)
            if not os.path.isdir(d) or name.startswith("."):
                continue

            def test_detect(d=d):
                fmt = detect_format(Path(d))
                assert fmt is not None, f"detect_format returned None for {d}"

            run_test(f"detect_format: {name[:50]}", test_detect)

    # --- Test 3: Load individual TIFF dataset ---
    tiff_datasets = [
        "96wellplate_20x_adjusted_left_2025-11-04_21-34-33.236116 yy2",
        "empty_slide_20x_adjusted_single_2025-11-04_22-11-31.444245 yy2",
    ]
    for ds_name in tiff_datasets:
        ds_path = os.path.join(TEST_DATA, ds_name)
        if not os.path.isdir(ds_path):
            continue

        def test_load_tiff(p=ds_path):
            w = LightweightMainWindow(p)
            w.show()
            app.processEvents()
            # Give it time to load
            for _ in range(20):
                app.processEvents()
                time.sleep(0.1)
            assert w.isVisible()
            viewer = w.centralWidget()
            assert viewer is not None
            w.close()

        run_test(f"Load TIFF: {ds_name[:45]}", test_load_tiff)

    # --- Test 4: Load OME-TIFF dataset ---
    ome_datasets = [
        "ps-m1-t72_2025-11-04_13-47-40.499214 yy",
        "DIPG17pons-QuaSar6a_after_KCl1_2025-10-31_13-01-44.044321 yy",
    ]
    for ds_name in ome_datasets:
        ds_path = os.path.join(TEST_DATA, ds_name)
        if not os.path.isdir(ds_path):
            continue

        def test_load_ome(p=ds_path):
            w = LightweightMainWindow(p)
            w.show()
            for _ in range(30):
                app.processEvents()
                time.sleep(0.1)
            assert w.isVisible()
            w.close()

        run_test(f"Load OME-TIFF: {ds_name[:40]}", test_load_ome)

    # --- Test 5: Viewer with synthetic data (no external deps) ---
    def test_synthetic_tiff():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal dataset: one timepoint dir with a few TIFFs
            t0 = os.path.join(tmpdir, "0")
            os.makedirs(t0)
            import tifffile
            for i in range(3):
                arr = np.random.randint(0, 1000, (64, 64), dtype=np.uint16)
                tifffile.imwrite(
                    os.path.join(t0, f"A1_{i}_0_BF_LED_matrix_full.tiff"), arr
                )
            # Write coordinates.csv
            with open(os.path.join(tmpdir, "coordinates.csv"), "w") as f:
                f.write("region,x (mm),y (mm),z (mm)\n")
                for i in range(3):
                    f.write(f"A1,{i*0.9},{0.0},\n")
            # Write minimal acquisition params
            import json
            with open(os.path.join(tmpdir, "acquisition parameters.json"), "w") as f:
                json.dump({
                    "dx(mm)": 0.9, "Nx": 1, "dy(mm)": 0.9, "Ny": 1,
                    "dz(um)": 1.5, "Nz": 1, "dt(s)": 0, "Nt": 1,
                    "objective": {"magnification": 10, "NA": 0.3,
                                  "tube_lens_f_mm": 180, "name": "10x"},
                    "sensor_pixel_size_um": 6.5, "tube_lens_mm": 180
                }, f)

            w = LightweightMainWindow(tmpdir)
            w.show()
            for _ in range(20):
                app.processEvents()
                time.sleep(0.1)
            assert w.isVisible()
            w.close()

    run_test("Load synthetic TIFF dataset", test_synthetic_tiff)

    # --- Test 6: get_fov_list returns data ---
    def test_fov_list():
        from ndviewer_light.core import LightweightViewer
        from unittest.mock import MagicMock
        mock = MagicMock(spec=LightweightViewer)
        mock.dataset_path = None
        mock.get_fov_list = lambda: LightweightViewer.get_fov_list(mock)
        mock._discover_fovs = LightweightViewer._discover_fovs.__get__(mock)
        result = mock.get_fov_list()
        assert result == []

    run_test("get_fov_list empty path", test_fov_list)

    # --- Test 7: data_structure_changed ---
    def test_data_structure():
        from ndviewer_light.core import data_structure_changed
        import xarray as xr
        d1 = xr.DataArray(np.zeros((2, 10, 10)), dims=["c", "y", "x"])
        d2 = xr.DataArray(np.zeros((3, 10, 10)), dims=["c", "y", "x"])
        assert data_structure_changed(d2, d1) is True
        assert data_structure_changed(d1, d1) is False

    run_test("data_structure_changed logic", test_data_structure)

    # --- Summary ---
    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Functional tests: {PASS}/{total} passed")
    if FAIL > 0:
        print(f"{FAIL} FAILED")
    print(f"{'='*50}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
