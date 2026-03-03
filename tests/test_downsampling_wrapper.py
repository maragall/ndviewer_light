"""Tests for Downsampling3DXarrayWrapper.

Tests verify that the custom DataWrapper correctly:
- Downsamples 3D volumes exceeding OpenGL texture limits
- Preserves full resolution for 2D slices
- Handles multichannel data correctly
- Leaves small volumes unchanged

Note: These tests require NDV to be available. They are skipped in CI
environments where NDV is not installed.
"""

import numpy as np
import pytest
import xarray as xr

# Import from ndviewer_light to register the wrapper (side effect of import)
from ndviewer_light import MAX_3D_TEXTURE_SIZE


@pytest.fixture
def data_wrapper_class():
    """Get the DataWrapper.create function."""
    from ndv.models._data_wrapper import DataWrapper

    return DataWrapper


class TestDownsampling3DXarrayWrapper:
    """Test suite for the Downsampling3DXarrayWrapper."""

    def test_wrapper_registration(self, data_wrapper_class):
        """Verify our wrapper is registered with higher priority."""
        from ndv.models._data_wrapper import _recurse_subclasses, DataWrapper

        wrappers = list(_recurse_subclasses(DataWrapper))
        wrapper_names = [w.__name__ for w in wrappers]

        assert "Downsampling3DXarrayWrapper" in wrapper_names

        # Check priority is higher (lower number) than XarrayWrapper
        our_wrapper = next(
            w for w in wrappers if w.__name__ == "Downsampling3DXarrayWrapper"
        )
        xarray_wrapper = next(w for w in wrappers if w.__name__ == "XarrayWrapper")
        assert our_wrapper.PRIORITY < xarray_wrapper.PRIORITY

    def test_wrapper_used_for_xarray(self, data_wrapper_class):
        """Verify our wrapper is used for xarray DataArrays."""
        data = xr.DataArray(
            np.zeros((10, 100, 100), dtype=np.uint16), dims=["z", "y", "x"]
        )
        wrapper = data_wrapper_class.create(data)
        assert type(wrapper).__name__ == "Downsampling3DXarrayWrapper"

    def test_3d_volume_downsampling(self, data_wrapper_class):
        """Test that large 3D volumes are downsampled."""
        large_size = MAX_3D_TEXTURE_SIZE + 1000  # 3048
        n_z = 10
        data = xr.DataArray(
            np.random.randint(0, 65536, (n_z, large_size, large_size), dtype=np.uint16),
            dims=["z", "y", "x"],  # Named 'z' so it's recognized as spatial
        )
        wrapper = data_wrapper_class.create(data)

        # Request full 3D volume
        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # Verify downsampled to within limits
        assert max(result.shape) <= MAX_3D_TEXTURE_SIZE
        assert result.dtype == np.uint16

    def test_2d_slice_full_resolution(self, data_wrapper_class):
        """Test that 2D slices remain at full resolution."""
        large_size = MAX_3D_TEXTURE_SIZE + 1000
        data = xr.DataArray(
            np.random.randint(0, 65536, (10, large_size, large_size), dtype=np.uint16),
            dims=["z", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        # Request single z-slice (2D)
        result = wrapper.isel({0: slice(0, 1), 1: slice(None), 2: slice(None)})
        squeezed = np.squeeze(result)

        # Verify full resolution preserved
        assert squeezed.shape == (large_size, large_size)

    def test_multichannel_single_channel_3d(self, data_wrapper_class):
        """Test multichannel data with single channel 3D request (NDV's pattern)."""
        large_size = MAX_3D_TEXTURE_SIZE + 1000
        n_channels = 4
        data = xr.DataArray(
            np.random.randint(
                0, 65536, (10, n_channels, large_size, large_size), dtype=np.uint16
            ),
            dims=["z", "channel", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        # Request single channel, all z (how NDV requests for composite 3D)
        result = wrapper.isel(
            {0: slice(None), 1: slice(0, 1), 2: slice(None), 3: slice(None)}
        )
        squeezed = np.squeeze(result)

        # Verify 3D and downsampled
        assert squeezed.ndim == 3
        assert max(squeezed.shape) <= MAX_3D_TEXTURE_SIZE

    def test_small_volume_unchanged(self, data_wrapper_class):
        """Test that small volumes are not modified."""
        small_size = MAX_3D_TEXTURE_SIZE - 500  # 1548
        n_z = 10
        data = xr.DataArray(
            np.random.randint(0, 65536, (n_z, small_size, small_size), dtype=np.uint16),
            dims=["z", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # Verify unchanged
        assert result.shape == (n_z, small_size, small_size)

    def test_multichannel_2d_not_downsampled(self, data_wrapper_class):
        """Test that multichannel 2D data (channel, y, x) is NOT downsampled."""
        large_size = MAX_3D_TEXTURE_SIZE + 1000
        n_channels = 4
        data = xr.DataArray(
            np.random.randint(
                0, 65536, (n_channels, large_size, large_size), dtype=np.uint16
            ),
            dims=["channel", "y", "x"],  # Named 'channel' so NOT treated as spatial
        )
        wrapper = data_wrapper_class.create(data)

        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # Verify NOT downsampled (multichannel 2D should stay full resolution)
        assert result.shape == (n_channels, large_size, large_size)

    def test_exactly_at_limit_unchanged(self, data_wrapper_class):
        """Test that volumes exactly at the limit are not modified."""
        n_z = 10
        data = xr.DataArray(
            np.random.randint(
                0,
                65536,
                (n_z, MAX_3D_TEXTURE_SIZE, MAX_3D_TEXTURE_SIZE),
                dtype=np.uint16,
            ),
            dims=["z", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # Verify unchanged
        assert result.shape == (n_z, MAX_3D_TEXTURE_SIZE, MAX_3D_TEXTURE_SIZE)

    def test_downsampling_preserves_dtype(self, data_wrapper_class):
        """Test that downsampling preserves the original dtype."""
        large_size = MAX_3D_TEXTURE_SIZE + 500
        n_z = 10
        data = xr.DataArray(
            np.random.randint(0, 65536, (n_z, large_size, large_size), dtype=np.uint16),
            dims=["z", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        assert result.dtype == np.uint16

    def test_downsampling_z_independent_xy_uniform(self, data_wrapper_class):
        """Test that z is scaled independently while xy preserves aspect ratio.

        - z: only downsampled if z itself exceeds the limit
        - x/y: use same scale factor (based on max) to preserve aspect ratio
        """
        n_z = 10
        data = xr.DataArray(
            np.random.randint(0, 65536, (n_z, 4000, 3000), dtype=np.uint16),
            dims=["z", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # z (10) doesn't exceed limit, should be unchanged
        assert result.shape[0] == n_z
        # xy uses uniform scale based on max(4000, 3000) = 4000
        # scale = 2048/4000 = 0.512
        # y: 4000 * 0.512 = 2048, x: 3000 * 0.512 = 1536
        # Allow small rounding differences from zoom-based resampling
        assert abs(result.shape[1] - MAX_3D_TEXTURE_SIZE) <= 2  # y scaled near limit
        assert result.shape[2] < MAX_3D_TEXTURE_SIZE  # x scaled proportionally

        # Verify aspect ratio is preserved
        original_ratio = 4000 / 3000
        result_ratio = result.shape[1] / result.shape[2]
        assert abs(original_ratio - result_ratio) < 0.1

    def test_large_z_downsampled_independently(self, data_wrapper_class):
        """Test that z is downsampled when z itself exceeds the texture limit."""
        large_z = MAX_3D_TEXTURE_SIZE + 500  # 2548 z slices
        small_xy = 512  # xy within limit
        data = xr.DataArray(
            np.zeros((large_z, small_xy, small_xy), dtype=np.uint16),
            dims=["z", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # z exceeds limit, should be scaled down close to MAX_3D_TEXTURE_SIZE
        # Allow small rounding differences from zoom-based resampling
        assert abs(result.shape[0] - MAX_3D_TEXTURE_SIZE) <= 1
        # xy doesn't exceed limit, should be unchanged
        assert result.shape[1] == small_xy
        assert result.shape[2] == small_xy

    def test_integer_indexing_drops_dimension(self, data_wrapper_class):
        """Test that integer indexing correctly drops dimensions.

        When isel() receives an integer index (not a slice), that dimension
        is dropped from the output. The zoom_factors should only include
        factors for remaining dimensions.
        """
        large_size = MAX_3D_TEXTURE_SIZE + 1000
        n_z = 10
        n_channels = 3
        data = xr.DataArray(
            np.random.randint(
                0, 65536, (n_z, n_channels, large_size, large_size), dtype=np.uint16
            ),
            dims=["z", "channel", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        # Use integer index for channel (drops that dimension)
        # Result should be 3D: (z, y, x)
        result = wrapper.isel({0: slice(None), 1: 0, 2: slice(None), 3: slice(None)})

        # Verify dimension was dropped
        assert result.ndim == 3
        # z unchanged, y/x downsampled
        # Allow small rounding differences from zoom-based resampling
        assert result.shape[0] == n_z
        assert abs(result.shape[1] - MAX_3D_TEXTURE_SIZE) <= 2
        assert abs(result.shape[2] - MAX_3D_TEXTURE_SIZE) <= 2

    def test_multichannel_all_channels_3d(self, data_wrapper_class):
        """Test 4D data (z, channel, y, x) with all channels requested.

        This tests the scenario when switching from 2D to 3D mode in NDV,
        where the result is 4D with channel in the middle. The channel
        dimension should be preserved while spatial dims are downsampled.
        """
        large_size = MAX_3D_TEXTURE_SIZE + 1000
        n_z = 5
        n_channels = 3
        data = xr.DataArray(
            np.random.randint(
                0, 65536, (n_z, n_channels, large_size, large_size), dtype=np.uint16
            ),
            dims=["z", "channel", "y", "x"],
        )
        wrapper = data_wrapper_class.create(data)

        # Request all z and all channels (how NDV requests when switching to 3D)
        result = wrapper.isel(
            {0: slice(None), 1: slice(None), 2: slice(None), 3: slice(None)}
        )

        # Verify 4D output with channels preserved, spatial downsampled
        assert result.ndim == 4
        assert result.shape[1] == n_channels  # channels preserved exactly
        assert max(result.shape[-2:]) <= MAX_3D_TEXTURE_SIZE  # y, x downsampled
        # z is also scaled (it's a spatial dimension)
        assert result.shape[0] <= n_z

    def test_pixel_sizes_in_attrs_preserved(self, data_wrapper_class):
        """Test that pixel size attrs are preserved through the wrapper.

        Physical pixel sizes (pixel_size_um, dz_um) are stored in attrs for
        reference and do not affect the numerical downsampling, which operates
        in index space; voxel scaling for display is handled separately via
        the vispy VolumeVisual patch in ndviewer_light.py.
        """
        large_size = MAX_3D_TEXTURE_SIZE + 500
        n_z = 50
        pixel_size_xy = 0.325  # µm per XY pixel
        pixel_size_z = 1.5  # µm per Z step

        data = xr.DataArray(
            np.random.randint(0, 65536, (n_z, large_size, large_size), dtype=np.uint16),
            dims=["z", "y", "x"],
            attrs={"pixel_size_um": pixel_size_xy, "dz_um": pixel_size_z},
        )
        wrapper = data_wrapper_class.create(data)

        # Verify attrs are accessible
        assert wrapper._data.attrs.get("pixel_size_um") == pixel_size_xy
        assert wrapper._data.attrs.get("dz_um") == pixel_size_z

        # Request full 3D volume
        result = wrapper.isel({0: slice(None), 1: slice(None), 2: slice(None)})

        # XY should be downsampled, Z unchanged (independent scaling)
        assert max(result.shape[-2:]) <= MAX_3D_TEXTURE_SIZE
        assert result.shape[0] == n_z  # Z not modified since under limit
