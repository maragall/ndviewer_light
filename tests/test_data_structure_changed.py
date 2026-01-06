"""
Unit tests for _data_structure_changed() method in LightweightViewer.

Tests cover:
1. Detection of dimension changes
2. Detection of channel count changes
3. Detection of channel name changes
4. Detection of LUT changes
5. Edge cases (None old_data, exceptions)

These tests verify the logic that determines when the NDV viewer needs
a full rebuild vs. an in-place data swap when switching datasets.
"""

import numpy as np
import pytest

# Import xarray - required for these tests
xr = pytest.importorskip("xarray")


def data_structure_changed(old_data, new_data) -> bool:
    """Standalone implementation of the detection logic for testing.

    This mirrors the logic in LightweightViewer._data_structure_changed()
    so we can test without instantiating the full viewer (which requires Qt).

    NOTE: This duplicates the logic from the actual implementation. If
    LightweightViewer._data_structure_changed() is modified, this function
    must be updated to match. When making changes to the detection logic,
    verify both implementations are in sync to avoid false test confidence.

    Args:
        old_data: Previous xarray DataArray (or None for first load)
        new_data: New xarray DataArray

    Returns:
        True if structure changed and viewer needs full rebuild.
    """
    if old_data is None:
        return True

    try:
        # Check if dims changed
        if old_data.dims != new_data.dims:
            return True

        # Check if channel count changed (use .sizes for cleaner access)
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
    except Exception:
        # On any error, assume structure changed to be safe
        return True


def create_test_xarray(
    dims=("time", "fov", "z", "channel", "y", "x"),
    shape=(1, 1, 1, 3, 100, 100),
    channel_names=None,
    luts=None,
):
    """Create a test xarray DataArray with specified structure.

    Args:
        dims: Tuple of dimension names
        shape: Tuple of dimension sizes (must match dims length)
        channel_names: List of channel names for attrs
        luts: Dict of channel index -> colormap name for attrs

    Returns:
        xr.DataArray with the specified structure
    """
    data = np.zeros(shape, dtype=np.uint16)

    # Build coords dict
    coords = {}
    for dim, size in zip(dims, shape):
        coords[dim] = list(range(size))

    arr = xr.DataArray(data, dims=dims, coords=coords)

    if channel_names is not None:
        arr.attrs["channel_names"] = channel_names
    if luts is not None:
        arr.attrs["luts"] = luts

    return arr


class TestDataStructureChanged:
    """Tests for _data_structure_changed() logic.

    These tests verify the detection logic without requiring Qt/NDV.
    We test the comparison logic directly using xarray DataArrays.
    """

    # --- Tests for None old_data ---

    def test_none_old_data_returns_true(self):
        """When old_data is None (first load), should return True."""
        new_data = create_test_xarray()
        assert data_structure_changed(None, new_data) is True

    # --- Tests for dimension changes ---

    def test_same_dims_returns_false(self):
        """When dims are identical, should not trigger rebuild for dims."""
        old = create_test_xarray(dims=("time", "fov", "z", "channel", "y", "x"))
        new = create_test_xarray(dims=("time", "fov", "z", "channel", "y", "x"))
        # Same dims, same channels, same attrs -> no change
        assert data_structure_changed(old, new) is False

    def test_different_dims_returns_true(self):
        """When dims differ, should return True."""
        old = create_test_xarray(dims=("time", "fov", "z", "channel", "y", "x"))
        new = create_test_xarray(
            dims=("time", "channel", "y", "x"),  # Missing fov and z
            shape=(1, 3, 100, 100),
        )
        assert data_structure_changed(old, new) is True

    def test_different_dim_order_returns_true(self):
        """When dim order differs, should return True."""
        old = create_test_xarray(dims=("time", "fov", "z", "channel", "y", "x"))
        new = create_test_xarray(
            dims=("time", "fov", "channel", "z", "y", "x"),  # z and channel swapped
            shape=(1, 1, 3, 1, 100, 100),
        )
        assert data_structure_changed(old, new) is True

    # --- Tests for channel count changes ---

    def test_fewer_channels_returns_true(self):
        """When new data has fewer channels, should return True."""
        old = create_test_xarray(shape=(1, 1, 1, 3, 100, 100))  # 3 channels
        new = create_test_xarray(shape=(1, 1, 1, 1, 100, 100))  # 1 channel
        assert data_structure_changed(old, new) is True

    def test_more_channels_returns_true(self):
        """When new data has more channels, should return True."""
        old = create_test_xarray(shape=(1, 1, 1, 1, 100, 100))  # 1 channel
        new = create_test_xarray(shape=(1, 1, 1, 4, 100, 100))  # 4 channels
        assert data_structure_changed(old, new) is True

    def test_same_channel_count_returns_false(self):
        """When channel count is same, should not trigger rebuild for count."""
        old = create_test_xarray(shape=(1, 1, 1, 3, 100, 100))
        new = create_test_xarray(shape=(1, 1, 1, 3, 100, 100))
        assert data_structure_changed(old, new) is False

    # --- Tests for channel name changes ---

    def test_different_channel_names_returns_true(self):
        """When channel names differ, should return True."""
        old = create_test_xarray(channel_names=["DAPI", "GFP", "RFP"])
        new = create_test_xarray(channel_names=["DAPI", "Cy5", "Cy7"])
        assert data_structure_changed(old, new) is True

    def test_same_channel_names_returns_false(self):
        """When channel names are identical, should not trigger rebuild."""
        old = create_test_xarray(channel_names=["DAPI", "GFP", "RFP"])
        new = create_test_xarray(channel_names=["DAPI", "GFP", "RFP"])
        assert data_structure_changed(old, new) is False

    def test_channel_names_added_returns_true(self):
        """When new data has channel names but old didn't, should return True."""
        old = create_test_xarray(channel_names=None)  # No channel_names attr
        new = create_test_xarray(channel_names=["DAPI", "GFP", "RFP"])
        assert data_structure_changed(old, new) is True

    def test_channel_names_removed_returns_true(self):
        """When new data lacks channel names but old had them, should return True."""
        old = create_test_xarray(channel_names=["DAPI", "GFP", "RFP"])
        new = create_test_xarray(channel_names=None)
        # old has names, new has [] (default) -> different
        assert data_structure_changed(old, new) is True

    # --- Tests for LUT changes ---

    def test_different_luts_returns_true(self):
        """When LUTs differ, should return True."""
        old = create_test_xarray(luts={0: "blue", 1: "green", 2: "red"})
        new = create_test_xarray(luts={0: "blue", 1: "yellow", 2: "magenta"})
        assert data_structure_changed(old, new) is True

    def test_same_luts_returns_false(self):
        """When LUTs are identical, should not trigger rebuild."""
        old = create_test_xarray(luts={0: "blue", 1: "green", 2: "red"})
        new = create_test_xarray(luts={0: "blue", 1: "green", 2: "red"})
        assert data_structure_changed(old, new) is False

    def test_luts_added_returns_true(self):
        """When new data has LUTs but old didn't, should return True."""
        old = create_test_xarray(luts=None)
        new = create_test_xarray(luts={0: "blue", 1: "green", 2: "red"})
        assert data_structure_changed(old, new) is True

    # --- Tests for spatial dimension changes (should NOT trigger rebuild) ---

    def test_different_xy_size_returns_false(self):
        """When only XY dimensions change, should NOT trigger rebuild.

        Spatial size changes don't require viewer rebuild - the NDV viewer
        handles different image sizes fine with in-place updates.
        """
        old = create_test_xarray(shape=(1, 1, 1, 3, 100, 100))
        new = create_test_xarray(shape=(1, 1, 1, 3, 200, 200))  # Larger XY
        assert data_structure_changed(old, new) is False

    def test_different_z_count_returns_false(self):
        """When only Z stack depth changes, should NOT trigger rebuild.

        More/fewer Z slices don't require viewer rebuild.
        """
        old = create_test_xarray(shape=(1, 1, 5, 3, 100, 100))  # 5 Z slices
        new = create_test_xarray(shape=(1, 1, 10, 3, 100, 100))  # 10 Z slices
        assert data_structure_changed(old, new) is False

    def test_different_time_count_returns_false(self):
        """When only timepoint count changes, should NOT trigger rebuild.

        This is the normal live acquisition case - more timepoints arrive.
        """
        old = create_test_xarray(shape=(5, 1, 1, 3, 100, 100))  # 5 timepoints
        new = create_test_xarray(shape=(10, 1, 1, 3, 100, 100))  # 10 timepoints
        assert data_structure_changed(old, new) is False

    # --- Tests for combined changes ---

    def test_channel_count_and_names_both_change(self):
        """When both channel count and names change, should return True."""
        old = create_test_xarray(
            shape=(1, 1, 1, 3, 100, 100),
            channel_names=["DAPI", "GFP", "RFP"],
        )
        new = create_test_xarray(
            shape=(1, 1, 1, 1, 100, 100),
            channel_names=["DAPI"],
        )
        assert data_structure_changed(old, new) is True

    def test_real_world_scenario_3ch_to_1ch(self):
        """Real-world: switching from 3-channel to 1-channel acquisition."""
        old = create_test_xarray(
            shape=(10, 4, 5, 3, 2048, 2048),
            channel_names=["405nm", "488nm", "561nm"],
            luts={0: "blue", 1: "green", 2: "red"},
        )
        new = create_test_xarray(
            shape=(1, 1, 1, 1, 2048, 2048),
            channel_names=["405nm"],
            luts={0: "blue"},
        )
        assert data_structure_changed(old, new) is True

    def test_real_world_scenario_same_channels_more_data(self):
        """Real-world: same channels, just more timepoints/FOVs (live acquisition)."""
        old = create_test_xarray(
            shape=(5, 2, 3, 3, 2048, 2048),
            channel_names=["405nm", "488nm", "561nm"],
            luts={0: "blue", 1: "green", 2: "red"},
        )
        new = create_test_xarray(
            shape=(10, 4, 3, 3, 2048, 2048),  # More time, more FOVs
            channel_names=["405nm", "488nm", "561nm"],
            luts={0: "blue", 1: "green", 2: "red"},
        )
        assert data_structure_changed(old, new) is False


class TestDataStructureChangedEdgeCases:
    """Edge case tests for _data_structure_changed()."""

    def test_empty_channel_names_both_sides(self):
        """When both have empty channel_names, should not trigger rebuild."""
        old = create_test_xarray(channel_names=[])
        new = create_test_xarray(channel_names=[])
        assert data_structure_changed(old, new) is False

    def test_no_channel_dim_both_sides(self):
        """When neither has channel dimension, should not trigger rebuild."""
        old = create_test_xarray(
            dims=("time", "y", "x"),
            shape=(1, 100, 100),
        )
        new = create_test_xarray(
            dims=("time", "y", "x"),
            shape=(5, 100, 100),
        )
        assert data_structure_changed(old, new) is False

    def test_luts_empty_dict_vs_none(self):
        """Empty luts dict should equal missing luts (both default to {})."""
        old = create_test_xarray(luts={})
        new = create_test_xarray(luts=None)
        # Both should resolve to {} via .get("luts", {})
        assert data_structure_changed(old, new) is False
