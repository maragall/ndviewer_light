"""Tests for IMA-191 push/live-acquisition FOV subset (N per region).

Two layers:
  1. Pure index math (kept_fov_indices / parse_fov_label).
  2. Viewer integration: a real LightweightViewer built offscreen, exercising
     the slider position <-> flat-index mapping used during live acquisition.

The subset remaps only the slider boundary; _current_fov_idx always stays the
true flat/global FOV index, so downstream loaders are unaffected. With the
subset off the mapping is identity (verified below), which is why the existing
push-mode tests continue to pass unchanged.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ndviewer_light.core import kept_fov_indices, parse_fov_label


# --- pure index math -------------------------------------------------------


def test_parse_fov_label():
    assert parse_fov_label("A1:3") == ("A1", 3)
    assert parse_fov_label("region_0:12") == ("region_0", 12)
    assert parse_fov_label("A1") == ("A1", 0)  # no suffix
    assert parse_fov_label("weird:x") == ("weird:x", 0)  # non-int suffix


def test_kept_disabled_is_identity():
    labels = ["A1:0", "A1:1", "A2:0"]
    assert kept_fov_indices(labels, 0) == [0, 1, 2]
    assert kept_fov_indices(labels, -1) == [0, 1, 2]


def test_kept_one_per_region():
    labels = ["A1:0", "A1:1", "A2:0", "A2:1", "A3:0"]
    assert kept_fov_indices(labels, 1) == [0, 2, 4]


def test_kept_n_per_region_mixed():
    labels = ["A1:0", "A1:1", "A1:2", "B2:0", "C3:0", "C3:1"]
    # keep up to 2 per region -> A1:{0,1}=idx0,1 ; B2:0=idx3 ; C3:{0,1}=idx4,5
    assert kept_fov_indices(labels, 2) == [0, 1, 3, 4, 5]


def test_kept_lexical_order_trap():
    # Labels arrive with fov 10 before fov 2 (as lexical discovery would order).
    labels = ["A1:0", "A1:1", "A1:10", "A1:11", "A1:2"]
    # first 3 by INTEGER fov -> fov 0,1,2 -> flat indices 0,1,4
    assert kept_fov_indices(labels, 3) == [0, 1, 4]


def test_kept_empty():
    assert kept_fov_indices([], 1) == []


# --- viewer integration ----------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def viewer(qapp):
    from ndviewer_light.core import LightweightViewer

    v = LightweightViewer()
    yield v
    v.close()


def _start(v, labels, n_z=1, h=32, w=32):
    v.start_acquisition(["BF LED matrix full"], n_z, h, w, labels)


def test_disabled_mapping_is_identity(viewer):
    _start(viewer, ["A1:0", "A1:1", "A2:0", "A2:1"])
    assert viewer._subset_enabled is False
    # identity: position == flat, slider max == flat max
    for flat in range(4):
        assert viewer._fov_pos_to_flat(flat) == flat
        assert viewer._fov_flat_to_pos(flat) == flat
    assert viewer._fov_slider_max_for(3) == 3


def test_subset_maps_positions_to_region_firsts(viewer):
    _start(viewer, ["A1:0", "A1:1", "A2:0", "A2:1", "A3:0"])
    viewer._subset_enabled = True
    viewer._subset_n_per_region = 1
    viewer._recompute_kept_fov_indices()
    assert viewer._subset_kept_indices == [0, 2, 4]
    # 3 kept positions map to flat 0, 2, 4
    assert [viewer._fov_pos_to_flat(p) for p in range(3)] == [0, 2, 4]
    # flat -> position, incl. a subset-out flat (1) snapping to nearest below
    assert viewer._fov_flat_to_pos(2) == 1
    assert viewer._fov_flat_to_pos(1) == 0  # A1:1 dropped -> nearest kept below


def test_slider_max_grows_only_for_kept_fovs(viewer):
    _start(viewer, ["A1:0", "A1:1", "A2:0", "A2:1"])
    viewer._subset_enabled = True
    viewer._subset_n_per_region = 1
    viewer._recompute_kept_fov_indices()  # kept = [0, 2]
    # As flat FOVs are acquired, slider position-max only advances at kept ones.
    assert viewer._fov_slider_max_for(0) == 0  # only flat 0 kept so far -> 1 pos
    assert viewer._fov_slider_max_for(1) == 0  # flat 1 dropped, still 1 kept
    assert viewer._fov_slider_max_for(2) == 1  # flat 2 kept -> 2 positions
    assert viewer._fov_slider_max_for(3) == 1  # flat 3 dropped


def test_go_to_subset_out_fov_loads_it_anyway(viewer):
    _start(viewer, ["A1:0", "A1:1", "A2:0", "A2:1"])
    viewer._subset_enabled = True
    viewer._subset_n_per_region = 1
    viewer._recompute_kept_fov_indices()  # kept = [0, 2]; A1:1 (flat 1) dropped
    ok = viewer.go_to_well_fov("A1", 1)
    assert ok is True
    # programmatic navigation wins: current flat index is the exact target
    assert viewer._current_fov_idx == 1


def test_toggle_during_acquisition_recomputes(viewer):
    _start(viewer, ["A1:0", "A1:1", "A2:0", "A2:1"])
    assert viewer._subset_kept_indices == [0, 1, 2, 3]  # identity while off
    viewer._on_subset_toggled(True)
    viewer._on_subset_n_changed(1)
    assert viewer._subset_kept_indices == [0, 2]
    viewer._on_subset_toggled(False)
    assert viewer._subset_kept_indices == [0, 1, 2, 3]  # back to identity


def test_subset_control_visible_only_multi_region(viewer):
    _start(viewer, ["A1:0", "A1:1"])  # single region
    assert viewer._subset_container.isVisibleTo(viewer) is False
    _start(viewer, ["A1:0", "A2:0"])  # two regions
    assert viewer._subset_container.isVisibleTo(viewer) is True
