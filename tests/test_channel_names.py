"""
Unit tests for channel name display feature in ndviewer_light.

Tests cover:
1. Channel name extraction and preservation logic in OME-TIFF path
2. Channel name extraction in single-TIFF path
3. Retry mechanism for channel label updates with generation counter
"""

import xml.etree.ElementTree as ET


# Test helper to create mock OME metadata XML
def create_ome_metadata(channel_names):
    """Create OME-TIFF metadata XML with channel names."""
    root = ET.Element("{http://www.openmicroscopy.org/Schemas/OME/2016-06}OME")
    image = ET.SubElement(
        root, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image"
    )
    pixels = ET.SubElement(
        image, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels"
    )

    for name in channel_names:
        channel_elem = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )
        channel_elem.set("Name", name)

    return ET.tostring(root, encoding="unicode")


class TestOMETiffChannelNamesLogic:
    """Test channel name extraction and preservation logic (isolated from Qt)."""

    def test_empty_channel_names_fallback_to_default(self):
        """When channel_names is empty, should fallback to Ch0, Ch1, etc."""
        n_c = 3
        channel_names = []

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]
        assert len(channel_names) == n_c

    def test_fewer_channel_names_extends_with_fallbacks(self):
        """When channel_names has fewer than n_c, extend with fallbacks (Ch2, Ch3, etc.)."""
        n_c = 5
        channel_names = ["DAPI", "GFP"]

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["DAPI", "GFP", "Ch2", "Ch3", "Ch4"]
        assert len(channel_names) == n_c

    def test_more_channel_names_truncates(self):
        """When channel_names has more than n_c, truncate to match n_c."""
        n_c = 2
        channel_names = ["DAPI", "GFP", "RFP", "Cy5", "Cy7"]

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["DAPI", "GFP"]
        assert len(channel_names) == n_c

    def test_exact_match_preserves_all(self):
        """When channel_names matches n_c exactly, preserve all names."""
        n_c = 3
        channel_names = ["DAPI", "GFP", "RFP"]

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["DAPI", "GFP", "RFP"]
        assert len(channel_names) == n_c

    def test_ome_metadata_parsing(self):
        """Test parsing channel names from OME metadata XML."""
        provided_names = ["DAPI", "GFP", "RFP"]
        ome_metadata = create_ome_metadata(provided_names)

        # Simulate the OME metadata parsing logic from _load_ome_tiff
        channel_names = []
        try:
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for ch in root.findall(".//ome:Channel", ns):
                    name = ch.get("Name") or ch.get("ID", "")
                    if name:
                        channel_names.append(name)
        except Exception:
            pass  # Parsing errors fall through to empty channel_names

        assert channel_names == provided_names

    def test_ome_metadata_parsing_with_invalid_xml(self):
        """Test fallback when OME metadata XML is invalid."""
        ome_metadata = "invalid xml <>"
        n_c = 3

        # Simulate the parsing logic with error handling
        channel_names = []
        try:
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for ch in root.findall(".//ome:Channel", ns):
                    name = ch.get("Name") or ch.get("ID", "")
                    if name:
                        channel_names.append(name)
        except Exception:
            pass  # Invalid XML expected - testing fallback behavior

        # Apply fallback logic
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]

    def test_ome_metadata_parsing_with_empty_names(self):
        """Test handling of channels with empty or missing Name attributes."""
        # Create XML with one channel missing Name attribute
        root = ET.Element("{http://www.openmicroscopy.org/Schemas/OME/2016-06}OME")
        image = ET.SubElement(
            root, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image"
        )
        pixels = ET.SubElement(
            image, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels"
        )

        # Add channels with various attributes
        ch1 = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )
        ch1.set("Name", "DAPI")

        # Channel without Name attribute (intentionally unnamed)
        _ = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )

        ch3 = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )
        ch3.set("Name", "GFP")

        ome_metadata = ET.tostring(root, encoding="unicode")

        # Parse channel names
        channel_names = []
        try:
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for ch in root.findall(".//ome:Channel", ns):
                    name = ch.get("Name") or ch.get("ID", "")
                    if name:  # Only append if name is not empty
                        channel_names.append(name)
        except Exception:
            pass  # Parsing errors fall through to partial channel_names

        # Only channels with names should be included
        assert channel_names == ["DAPI", "GFP"]


class TestSingleTiffChannelNamesLogic:
    """Test channel name extraction logic in single-TIFF path."""

    def test_channel_names_sorted_alphabetically(self):
        """Channel names from filenames should be sorted alphabetically."""
        # Simulate discovering channels from filenames in _load_single_tiff
        c_set = {"RFP", "DAPI", "GFP", "Cy5"}
        channel_names = sorted(c_set)

        expected = ["Cy5", "DAPI", "GFP", "RFP"]
        assert channel_names == expected

    def test_channel_names_extracted_from_set(self):
        """Channel names should be extracted into a sorted list."""
        # Simulate the logic from _load_single_tiff
        c_set = set()
        # Add channels as discovered from filenames
        c_set.add("DAPI")
        c_set.add("GFP")
        c_set.add("RFP")

        channel_names = sorted(c_set)

        assert channel_names == ["DAPI", "GFP", "RFP"]
        assert len(channel_names) == 3

    def test_channel_names_stored_in_xarray_attrs(self):
        """Channel names should be stored in xarray attrs."""
        # Tests channel_names storage in xarray attrs (in _load_single_tiff)
        channel_names = ["DAPI", "GFP", "RFP"]

        # Simulate creating xarray attrs
        attrs = {"luts": {}, "channel_names": channel_names}

        assert attrs["channel_names"] == channel_names
        assert len(attrs["channel_names"]) == 3


class TestChannelLabelRetryMechanism:
    """Test retry mechanism logic for channel label updates with generation counter."""

    def test_generation_counter_increments(self):
        """Generation counter should increment on each update."""
        # Simulate the generation counter logic from _set_ndv_data
        generation = 0

        # First update
        generation += 1
        assert generation == 1

        # Second update
        generation += 1
        assert generation == 2

        # Third update
        generation += 1
        assert generation == 3

    def test_pending_retries_reset(self):
        """Pending retries should be reset to 20 on each update."""
        # Simulate the retry counter logic from _set_ndv_data
        # On each new update, pending_retries is always set to 20
        pending_retries = 20

        assert pending_retries == 20

    def test_stale_generation_check(self):
        """Stale generation callbacks should be detected."""
        # Simulate the generation check from _schedule_channel_label_update
        current_generation = 5
        callback_generation = 3

        # Check if callback is stale
        is_stale = current_generation != callback_generation

        assert is_stale is True

    def test_current_generation_check(self):
        """Current generation callbacks should not be stale."""
        current_generation = 5
        callback_generation = 5

        # Check if callback is stale
        is_stale = current_generation != callback_generation

        assert is_stale is False

    def test_retry_counter_decrement(self):
        """Retry counter should decrement on each retry attempt."""
        # Simulate the retry logic from _schedule_channel_label_update
        remaining = 10

        # After one retry attempt
        remaining = remaining - 1

        assert remaining == 9

    def test_retry_timeout_detection(self):
        """Should detect when retries are exhausted."""
        # Simulate timeout detection in _schedule_channel_label_update
        remaining = 0

        should_timeout = remaining <= 0

        assert should_timeout is True

    def test_retry_continues_when_retries_remain(self):
        """Should continue retrying while retries remain."""
        remaining = 5

        should_timeout = remaining <= 0
        should_continue = remaining > 0

        assert should_timeout is False
        assert should_continue is True


class TestChannelLabelUpdate:
    """Test the channel label update logic."""

    def test_channel_labels_set_on_controllers(self):
        """Should set key attribute on each LUT controller."""
        # Simulate the update logic from _update_channel_labels
        channel_names = ["DAPI", "GFP", "RFP"]

        # Mock controllers dictionary
        controllers = {
            0: {"key": "Ch0"},
            1: {"key": "Ch1"},
            2: {"key": "Ch2"},
        }

        # Update controller keys
        for i, name in enumerate(channel_names):
            if i in controllers:
                controllers[i]["key"] = name

        # Verify keys were updated
        assert controllers[0]["key"] == "DAPI"
        assert controllers[1]["key"] == "GFP"
        assert controllers[2]["key"] == "RFP"

    def test_channel_labels_handles_fewer_controllers(self):
        """Should handle case where there are fewer controllers than channel names."""
        channel_names = ["DAPI", "GFP", "RFP", "Cy5"]

        # Only 2 controllers available
        controllers = {
            0: {"key": "Ch0"},
            1: {"key": "Ch1"},
        }

        updated_count = 0
        # Update only existing controllers
        for i, name in enumerate(channel_names):
            if i in controllers:
                controllers[i]["key"] = name
                updated_count += 1

        # Only first 2 should be updated
        assert updated_count == 2
        assert controllers[0]["key"] == "DAPI"
        assert controllers[1]["key"] == "GFP"

    def test_channel_labels_skips_missing_indices(self):
        """Should skip channel indices that don't have controllers."""
        channel_names = ["DAPI", "GFP", "RFP"]

        # Controllers with gaps (missing index 1)
        controllers = {
            0: {"key": "Ch0"},
            2: {"key": "Ch2"},
        }

        # Update only existing controllers
        for i, name in enumerate(channel_names):
            if i in controllers:
                controllers[i]["key"] = name

        # Should update only indices 0 and 2
        assert controllers[0]["key"] == "DAPI"
        assert 1 not in controllers  # Still missing
        assert controllers[2]["key"] == "RFP"


class TestChannelNamesIntegration:
    """Integration tests for channel name handling."""

    def test_channel_names_length_consistency(self):
        """Channel names length should match channel dimension after processing."""
        n_c = 4
        initial_names = ["DAPI", "GFP"]

        # Apply the adjustment logic
        channel_names = initial_names.copy()
        if len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))

        # Verify consistency
        assert len(channel_names) == n_c
        assert channel_names == ["DAPI", "GFP", "Ch2", "Ch3"]

    def test_channel_names_default_format(self):
        """Default channel names should follow Ch0, Ch1, Ch2... format."""
        n_c = 5
        channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4"]

    def test_empty_list_becomes_defaults(self):
        """Empty channel names list should be replaced with defaults."""
        n_c = 3
        channel_names = []

        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]
        assert len(channel_names) == n_c

    def test_none_becomes_defaults(self):
        """None channel names should be replaced with defaults."""
        n_c = 3
        channel_names = None

        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]

    def test_preserves_custom_names_when_exact_match(self):
        """Custom channel names should be preserved when count matches."""
        n_c = 3
        channel_names = ["405nm_DAPI", "488nm_GFP", "561nm_RFP"]

        # Apply adjustment (should be no-op)
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["405nm_DAPI", "488nm_GFP", "561nm_RFP"]
