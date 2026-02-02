"""Tests for the push-based zarr API.

Tests verify the API contract without requiring Qt initialization.
These are logic-focused tests that verify state management and signal handling.
"""

from pathlib import Path


class TestZarrPushApiLogic:
    """Test suite for zarr push API logic (no Qt required)."""

    def test_channel_map_creation(self):
        """Test that channel name to index mapping is created correctly."""
        channels = ["DAPI", "GFP", "RFP"]
        channel_map = {name: i for i, name in enumerate(channels)}

        assert channel_map["DAPI"] == 0
        assert channel_map["GFP"] == 1
        assert channel_map["RFP"] == 2

    def test_channel_lookup(self):
        """Test looking up channel index from name."""
        channel_map = {"DAPI": 0, "GFP": 1, "RFP": 2}

        # Valid lookups
        assert channel_map.get("DAPI", -1) == 0
        assert channel_map.get("GFP", -1) == 1
        assert channel_map.get("RFP", -1) == 2

        # Invalid lookup returns -1
        assert channel_map.get("Unknown", -1) == -1

    def test_fov_label_format(self):
        """Test FOV label format used by Squid."""
        fov_labels = ["A1:0", "A1:1", "A2:0", "A2:1"]

        # Labels should be parseable as well:fov
        for label in fov_labels:
            well, fov = label.split(":")
            assert len(well) >= 2  # At least row + column
            assert fov.isdigit()

    def test_max_fov_per_time_tracking(self):
        """Test tracking max FOV index per timepoint."""
        max_fov_per_time = {}

        # Simulate frame registrations
        frames = [
            (0, 0),  # t=0, fov=0
            (0, 1),  # t=0, fov=1
            (0, 2),  # t=0, fov=2
            (1, 0),  # t=1, fov=0
            (1, 1),  # t=1, fov=1
        ]

        for t, fov_idx in frames:
            current_max = max_fov_per_time.get(t, -1)
            if fov_idx > current_max:
                max_fov_per_time[t] = fov_idx

        assert max_fov_per_time[0] == 2
        assert max_fov_per_time[1] == 1

    def test_max_time_tracking(self):
        """Test tracking maximum timepoint."""
        max_time_idx = 0

        frames = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]

        for t, _ in frames:
            max_time_idx = max(max_time_idx, t)

        assert max_time_idx == 2


class TestZarrCacheKey:
    """Test zarr cache key generation."""

    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different planes."""
        keys = set()

        for t in range(2):
            for fov in range(2):
                for z in range(3):
                    for ch in range(2):
                        key = ("zarr", t, fov, z, ch)
                        assert key not in keys, f"Duplicate key: {key}"
                        keys.add(key)

        # Should have 2*2*3*2 = 24 unique keys
        assert len(keys) == 24

    def test_cache_key_format(self):
        """Test cache key tuple format."""
        key = ("zarr", 0, 1, 2, 3)

        assert key[0] == "zarr"  # Prefix to distinguish from TIFF cache
        assert key[1] == 0  # t
        assert key[2] == 1  # fov
        assert key[3] == 2  # z
        assert key[4] == 3  # channel


class TestZarrMetadataIntegration:
    """Test integration between metadata parsing and push API."""

    def test_channel_colors_to_luts(self):
        """Test converting channel colors to LUTs."""
        from ndviewer_light import hex_to_colormap, wavelength_to_colormap
        from ndviewer_light.core import extract_wavelength

        channel_names = ["DAPI", "GFP", "RFP"]
        channel_colors = ["0000FF", "00FF00", "FF0000"]

        luts = {}
        for i, name in enumerate(channel_names):
            if i < len(channel_colors) and channel_colors[i]:
                luts[i] = hex_to_colormap(channel_colors[i])
            else:
                luts[i] = wavelength_to_colormap(extract_wavelength(name))

        assert luts[0] == "blue"  # DAPI
        assert luts[1] == "green"  # GFP
        assert luts[2] == "red"  # RFP

    def test_fallback_to_wavelength_colormap(self):
        """Test fallback to wavelength-based colormap when no colors provided."""
        from ndviewer_light import wavelength_to_colormap
        from ndviewer_light.core import extract_wavelength

        channel_names = ["Fluorescence 488 nm Ex", "Fluorescence 561 nm Ex"]

        luts = {}
        for i, name in enumerate(channel_names):
            luts[i] = wavelength_to_colormap(extract_wavelength(name))

        assert luts[0] == "green"  # 488nm
        assert luts[1] == "yellow"  # 561nm


class TestZarrStateManagement:
    """Test state management for zarr acquisition."""

    def test_acquisition_state_transitions(self):
        """Test state transitions during acquisition lifecycle."""
        # Initial state
        zarr_acquisition_active = False
        zarr_acquisition_path = None

        # Start acquisition
        zarr_acquisition_active = True
        zarr_acquisition_path = Path("/tmp/test.zarr")

        assert zarr_acquisition_active is True
        assert zarr_acquisition_path is not None

        # End acquisition
        zarr_acquisition_active = False

        assert zarr_acquisition_active is False
        # Path preserved for browsing
        assert zarr_acquisition_path is not None

    def test_push_mode_detection(self):
        """Test is_push_mode_active logic."""
        fov_labels = []
        zarr_acquisition_active = False

        # Neither active
        assert not (bool(fov_labels) or zarr_acquisition_active)

        # FOV labels set (TIFF push mode)
        fov_labels = ["A1:0", "A1:1"]
        assert bool(fov_labels) or zarr_acquisition_active

        # Zarr mode active
        fov_labels = []
        zarr_acquisition_active = True
        assert bool(fov_labels) or zarr_acquisition_active

        # Both active
        fov_labels = ["A1:0"]
        zarr_acquisition_active = True
        assert bool(fov_labels) or zarr_acquisition_active


class TestZarrApiValidation:
    """Test API validation for zarr acquisition methods."""

    def test_empty_fov_paths_validation(self):
        """Test that empty fov_paths raises ValueError.

        This validates the API contract that start_zarr_acquisition()
        requires at least one FOV path.
        """

        # Simulate the validation logic from start_zarr_acquisition()
        def validate_fov_paths(fov_paths):
            if not fov_paths:
                raise ValueError("fov_paths must not be empty")

        # Empty list should raise
        import pytest

        with pytest.raises(ValueError, match="fov_paths must not be empty"):
            validate_fov_paths([])

        # None should raise
        with pytest.raises(ValueError, match="fov_paths must not be empty"):
            validate_fov_paths(None)

        # Non-empty list should pass
        validate_fov_paths(["/path/to/fov.zarr"])  # No exception

    def test_fov_paths_labels_mismatch_truncation(self):
        """Test that mismatched fov_paths/fov_labels are truncated to shorter."""
        fov_paths = ["/path/fov0.zarr", "/path/fov1.zarr", "/path/fov2.zarr"]
        fov_labels = ["A1:0", "A1:1"]  # Only 2 labels for 3 paths

        # Simulate truncation logic from start_zarr_acquisition()
        if len(fov_paths) != len(fov_labels):
            min_len = min(len(fov_paths), len(fov_labels))
            fov_paths = list(fov_paths)[:min_len]
            fov_labels = list(fov_labels)[:min_len]

        assert len(fov_paths) == 2
        assert len(fov_labels) == 2
        assert fov_paths == ["/path/fov0.zarr", "/path/fov1.zarr"]


class TestMultiRegion6D:
    """Test multi-region 6D zarr support (6d_regions mode)."""

    def test_global_fov_index_calculation(self):
        """Test converting global FOV index to (region_idx, local_fov_idx)."""
        # Simulate _global_to_region_fov logic
        fovs_per_region = [4, 6, 3]  # 3 regions with 4, 6, 3 FOVs
        region_fov_offsets = [0, 4, 10]  # cumulative: [0, 0+4, 0+4+6]
        total_fovs = sum(fovs_per_region)  # 13

        def global_to_region_fov(global_fov_idx):
            for region_idx, offset in enumerate(region_fov_offsets):
                next_offset = (
                    region_fov_offsets[region_idx + 1]
                    if region_idx + 1 < len(region_fov_offsets)
                    else total_fovs
                )
                if offset <= global_fov_idx < next_offset:
                    return region_idx, global_fov_idx - offset
            return 0, global_fov_idx

        # Region 0: global FOV 0-3 → local FOV 0-3
        assert global_to_region_fov(0) == (0, 0)
        assert global_to_region_fov(1) == (0, 1)
        assert global_to_region_fov(3) == (0, 3)

        # Region 1: global FOV 4-9 → local FOV 0-5
        assert global_to_region_fov(4) == (1, 0)
        assert global_to_region_fov(5) == (1, 1)
        assert global_to_region_fov(9) == (1, 5)

        # Region 2: global FOV 10-12 → local FOV 0-2
        assert global_to_region_fov(10) == (2, 0)
        assert global_to_region_fov(11) == (2, 1)
        assert global_to_region_fov(12) == (2, 2)

    def test_region_fov_offsets_computation(self):
        """Test computing cumulative FOV offsets from fovs_per_region."""
        fovs_per_region = [4, 6, 3]

        region_fov_offsets = []
        offset = 0
        for n_fov in fovs_per_region:
            region_fov_offsets.append(offset)
            offset += n_fov

        assert region_fov_offsets == [0, 4, 10]
        assert sum(fovs_per_region) == 13

    def test_fov_labels_generation(self):
        """Test generating flattened FOV labels for multi-region mode."""
        region_labels = ["region_0", "region_1", "region_2"]
        fovs_per_region = [4, 6, 3]

        fov_labels = []
        for region_label, n_fov in zip(region_labels, fovs_per_region):
            for fov_in_region in range(n_fov):
                fov_labels.append(f"{region_label}:{fov_in_region}")

        # Should have 13 labels total
        assert len(fov_labels) == 13

        # Check first region labels
        assert fov_labels[0] == "region_0:0"
        assert fov_labels[3] == "region_0:3"

        # Check second region labels
        assert fov_labels[4] == "region_1:0"
        assert fov_labels[9] == "region_1:5"

        # Check third region labels
        assert fov_labels[10] == "region_2:0"
        assert fov_labels[12] == "region_2:2"

    def test_fov_labels_with_variable_fovs(self):
        """Test FOV labels with different FOV counts per region."""
        region_labels = ["scan_A", "scan_B"]
        fovs_per_region = [2, 5]

        fov_labels = []
        for region_label, n_fov in zip(region_labels, fovs_per_region):
            for fov_in_region in range(n_fov):
                fov_labels.append(f"{region_label}:{fov_in_region}")

        expected = [
            "scan_A:0",
            "scan_A:1",
            "scan_B:0",
            "scan_B:1",
            "scan_B:2",
            "scan_B:3",
            "scan_B:4",
        ]
        assert fov_labels == expected

    def test_global_fov_to_region_boundary_cases(self):
        """Test boundary cases in global FOV conversion."""
        # Single FOV per region: fovs_per_region = [1, 1, 1]
        region_fov_offsets = [0, 1, 2]
        total_fovs = 3

        def global_to_region_fov(global_fov_idx):
            for region_idx, offset in enumerate(region_fov_offsets):
                next_offset = (
                    region_fov_offsets[region_idx + 1]
                    if region_idx + 1 < len(region_fov_offsets)
                    else total_fovs
                )
                if offset <= global_fov_idx < next_offset:
                    return region_idx, global_fov_idx - offset
            return 0, global_fov_idx

        assert global_to_region_fov(0) == (0, 0)
        assert global_to_region_fov(1) == (1, 0)
        assert global_to_region_fov(2) == (2, 0)

    def test_notify_frame_global_fov_conversion(self):
        """Test that notify_zarr_frame correctly converts region_idx + fov_idx to global."""
        # Simulate the conversion logic in notify_zarr_frame
        region_fov_offsets = [0, 4, 10]
        zarr_6d_regions_mode = True

        def compute_global_fov(fov_idx, region_idx):
            if zarr_6d_regions_mode and region_fov_offsets:
                if region_idx < len(region_fov_offsets):
                    return region_fov_offsets[region_idx] + fov_idx
            return fov_idx

        # Region 0, FOV 2 → global FOV 2
        assert compute_global_fov(2, 0) == 2

        # Region 1, FOV 3 → global FOV 7 (4 + 3)
        assert compute_global_fov(3, 1) == 7

        # Region 2, FOV 1 → global FOV 11 (10 + 1)
        assert compute_global_fov(1, 2) == 11

    def test_6d_regions_push_mode_detection(self):
        """Test is_zarr_push_mode_active includes 6d_regions mode."""
        # Simulate is_zarr_push_mode_active logic (matches core.py implementation)
        zarr_acquisition_active = False
        zarr_fov_paths = []
        zarr_6d_regions_mode = False

        def is_zarr_push_mode_active():
            return (
                zarr_acquisition_active or bool(zarr_fov_paths) or zarr_6d_regions_mode
            )

        # None active
        assert not is_zarr_push_mode_active()

        # Only 6d_regions mode active
        zarr_6d_regions_mode = True
        assert is_zarr_push_mode_active()

        # Reset and check acquisition active
        zarr_6d_regions_mode = False
        zarr_acquisition_active = True
        assert is_zarr_push_mode_active()

        # Reset and check fov_paths active
        zarr_acquisition_active = False
        zarr_fov_paths = ["/path/to/fov.zarr"]
        assert is_zarr_push_mode_active()


class TestCachingBehavior:
    """Test plane caching behavior during live acquisition."""

    def test_written_plane_tracking(self):
        """Test that only written planes are cached during live acquisition.

        Uses a set to track which planes have been written via notify_zarr_frame().
        This is O(1) lookup vs O(n) plane.max() check, and handles legitimately
        black images correctly.
        """
        # Simulate the written planes tracking
        written_planes = set()
        acquisition_active = True

        def should_cache(cache_key):
            """Replicate the caching logic: cache if not live or if written."""
            if not acquisition_active:
                return True  # Browsing existing data - cache everything
            return cache_key in written_planes

        # During live acquisition, unwritten plane should not be cached
        key1 = ("zarr", 0, 4, 0, 1)  # t=0, fov=4, z=0, ch=1
        assert not should_cache(key1)

        # After notify_zarr_frame(), plane should be cached
        written_planes.add(key1)
        assert should_cache(key1)

        # Different plane still not cached until written
        key2 = ("zarr", 0, 5, 0, 1)
        assert not should_cache(key2)

        # When acquisition ends, all planes cached (browsing mode)
        acquisition_active = False
        assert should_cache(key2)  # Now cacheable even though not in written set

    def test_race_condition_prevention(self):
        """Test that was_written_before_read prevents caching stale data.

        Race condition scenario:
        1. Viewer starts reading plane (sees zeros, not yet written)
        2. During read, notify_zarr_frame() is called (plane now marked written)
        3. Read completes with stale zeros
        4. Old code: cache_key in written_planes → True → caches zeros (BUG)
        5. New code: was_written_before_read is False → don't cache (CORRECT)
        """
        written_planes = set()
        acquisition_active = True

        def should_cache_with_race_prevention(cache_key, was_written_before_read):
            """New caching logic that uses pre-read state."""
            if not acquisition_active:
                return True
            return was_written_before_read

        key = ("zarr", 0, 0, 0, 0)

        # Simulate race condition:
        # 1. Check written state BEFORE read
        was_written_before_read = key in written_planes  # False

        # 2. During read, notify_zarr_frame() is called
        written_planes.add(key)

        # 3. Read completes - should we cache?
        # Old logic would check: key in written_planes → True → cache stale data
        # New logic uses was_written_before_read → False → don't cache
        assert not should_cache_with_race_prevention(key, was_written_before_read)

        # Next read: plane is already written, so was_written_before_read is True
        was_written_before_read = key in written_planes  # True now
        assert should_cache_with_race_prevention(key, was_written_before_read)

    def test_cache_invalidation_on_notify(self):
        """Test that cache entries are invalidated when notify_zarr_frame() is called.

        This handles the case where stale data was cached before the race
        condition prevention was in place, or when a plane is re-written.
        """
        # Simulate cache with stale data
        cache = {}
        cache_key = ("zarr", 0, 0, 0, 0)
        cache[cache_key] = "stale_zeros"

        def invalidate(key):
            """Simulate MemoryBoundedLRUCache.invalidate()."""
            if key in cache:
                del cache[key]
                return True
            return False

        # Stale data is in cache
        assert cache_key in cache

        # notify_zarr_frame() calls invalidate()
        result = invalidate(cache_key)
        assert result is True
        assert cache_key not in cache

        # Invalidating non-existent key returns False
        result = invalidate(("zarr", 1, 1, 1, 1))
        assert result is False


class TestMemoryBoundedLRUCacheInvalidate:
    """Test MemoryBoundedLRUCache.invalidate() method."""

    def test_invalidate_existing_entry(self):
        """Test invalidating an entry that exists in cache."""
        import numpy as np

        from ndviewer_light import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=10 * 1024 * 1024)
        key = ("zarr", 0, 0, 0, 0)
        plane = np.zeros((100, 100), dtype=np.uint16)

        cache.put(key, plane)
        assert key in cache
        assert len(cache) == 1

        result = cache.invalidate(key)
        assert result is True
        assert key not in cache
        assert len(cache) == 0

    def test_invalidate_nonexistent_entry(self):
        """Test invalidating an entry that doesn't exist."""
        from ndviewer_light import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=10 * 1024 * 1024)
        key = ("zarr", 0, 0, 0, 0)

        result = cache.invalidate(key)
        assert result is False
        assert len(cache) == 0

    def test_invalidate_updates_memory_tracking(self):
        """Test that invalidate() correctly updates memory accounting."""
        import numpy as np

        from ndviewer_light import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=10 * 1024 * 1024)

        # Add two entries
        key1 = ("zarr", 0, 0, 0, 0)
        key2 = ("zarr", 0, 0, 0, 1)
        plane1 = np.zeros((100, 100), dtype=np.uint16)  # 20KB
        plane2 = np.zeros((100, 100), dtype=np.uint16)  # 20KB

        cache.put(key1, plane1)
        cache.put(key2, plane2)

        initial_memory = cache._current_memory
        assert initial_memory == plane1.nbytes + plane2.nbytes

        # Invalidate one entry
        cache.invalidate(key1)

        # Memory should be reduced
        assert cache._current_memory == plane2.nbytes
        assert cache._current_memory == initial_memory - plane1.nbytes

    def test_invalidate_thread_safety(self):
        """Test that invalidate() is thread-safe."""
        import threading

        import numpy as np

        from ndviewer_light import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=10 * 1024 * 1024)
        errors = []

        def add_and_invalidate(thread_id):
            try:
                for i in range(100):
                    key = ("zarr", thread_id, i, 0, 0)
                    plane = np.zeros((10, 10), dtype=np.uint16)
                    cache.put(key, plane)
                    cache.invalidate(key)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_and_invalidate, args=(i,)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestWriteZarrMetadata:
    """Tests for _write_zarr_metadata function in simulate_zarr_acquisition.py."""

    def test_merge_into_existing_preserves_array_structure(self, tmp_path):
        """Test that merge_into_existing=True preserves existing zarr.json structure."""
        import json
        import sys

        # Import the function from simulate script
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from simulate_zarr_acquisition import _write_zarr_metadata

        zarr_path = tmp_path / "test.zarr"
        zarr_path.mkdir()

        # Create an existing array zarr.json (simulating what tensorstore creates)
        existing_zarr_json = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4, 10, 3, 5, 512, 512],
            "data_type": "uint16",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [1, 1, 1, 1, 512, 512]},
            },
            "chunk_key_encoding": {"name": "default"},
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }
        with open(zarr_path / "zarr.json", "w") as f:
            json.dump(existing_zarr_json, f)

        # Call with merge_into_existing=True
        _write_zarr_metadata(
            zarr_path,
            channels=["DAPI", "GFP", "RFP"],
            channel_colors=["#0000FF", "#00FF00", "#FF0000"],
            pixel_size_um=0.25,
            z_step_um=1.0,
            merge_into_existing=True,
        )

        # Read the result
        with open(zarr_path / "zarr.json", "r") as f:
            result = json.load(f)

        # Should preserve array structure
        assert result["zarr_format"] == 3
        assert result["node_type"] == "array"
        assert result["shape"] == [4, 10, 3, 5, 512, 512]
        assert result["data_type"] == "uint16"

        # Should have added attributes
        assert "attributes" in result
        assert "ome" in result["attributes"]
        assert "multiscales" in result["attributes"]["ome"]

        # Dataset path should be "." for merged (6D at root)
        datasets = result["attributes"]["ome"]["multiscales"][0]["datasets"]
        assert datasets[0]["path"] == "."

    def test_non_merge_creates_group_structure(self, tmp_path):
        """Test that merge_into_existing=False creates a group zarr.json."""
        import json
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from simulate_zarr_acquisition import _write_zarr_metadata

        zarr_path = tmp_path / "test.zarr"
        zarr_path.mkdir()

        # Call without merge (default behavior)
        _write_zarr_metadata(
            zarr_path,
            channels=["DAPI", "GFP"],
            channel_colors=["#0000FF", "#00FF00"],
            pixel_size_um=0.25,
            z_step_um=1.0,
            merge_into_existing=False,
        )

        # Read the result
        with open(zarr_path / "zarr.json", "r") as f:
            result = json.load(f)

        # Should be a group (not array)
        assert result["zarr_format"] == 3
        assert result["node_type"] == "group"

        # Dataset path should be "0" for non-merged (array at /0 subdirectory)
        datasets = result["attributes"]["ome"]["multiscales"][0]["datasets"]
        assert datasets[0]["path"] == "0"
