# Test Suite for ndviewer_light

This directory contains unit tests for the ndviewer_light application.

## Running Tests

Ensure pytest is installed in your environment:
```bash
pip install pytest
```

Run all tests:
```bash
pytest tests/
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_channel_names.py -v
```

Run specific test class:
```bash
pytest tests/test_channel_names.py::TestOMETiffChannelNamesLogic -v
```

Run specific test:
```bash
pytest tests/test_channel_names.py::TestOMETiffChannelNamesLogic::test_empty_channel_names_fallback_to_default -v
```

## Test Files

### test_channel_names.py

Comprehensive unit tests for the channel name display feature. Tests cover five main areas:

#### 1. OME-TIFF Channel Name Logic (`TestOMETiffChannelNamesLogic`)

Tests the channel name extraction and preservation logic from OME-TIFF metadata (in `_load_ome_tiff` method):

- **Empty channel names fallback**: When no channel names are found in metadata, defaults to `Ch0`, `Ch1`, `Ch2`, etc.
- **Extension with fallbacks**: When fewer channel names than channels (e.g., 2 names for 5 channels), extends with `Ch2`, `Ch3`, `Ch4`
- **Truncation**: When more channel names than channels (e.g., 5 names for 2 channels), truncates to match
- **Exact match preservation**: When channel names count matches exactly, preserves all names
- **XML metadata parsing**: Tests parsing of OME-TIFF XML metadata with proper namespace handling
- **Error handling**: Tests fallback behavior when XML is invalid or channels have missing Name attributes

#### 2. Single-TIFF Channel Name Logic (`TestSingleTiffChannelNamesLogic`)

Tests channel name extraction from single-TIFF filenames (in `_load_single_tiff` method):

- **Alphabetical sorting**: Channel names extracted from filenames are sorted alphabetically
- **Set to list conversion**: Tests the conversion from set (during discovery) to sorted list
- **Storage in attrs**: Verifies channel names are properly stored in xarray attrs

#### 3. Retry Mechanism (`TestChannelLabelRetryMechanism`)

Tests the retry mechanism for updating channel labels in the NDV viewer (in `_set_ndv_data` and `_schedule_channel_label_update` methods):

- **Generation counter**: Tests that generation counter increments with each update
- **Pending retries initialization**: Tests that retry counter is reset to 20 on each update
- **Stale callback detection**: Tests that callbacks with old generation numbers are detected and ignored
- **Current generation processing**: Tests that callbacks with current generation proceed normally
- **Retry decrement**: Tests that retry counter decrements on each attempt
- **Timeout detection**: Tests that retries stop when counter reaches 0

The generation counter mechanism is critical for preventing stale callbacks from previous data loads from interfering with new loads. When a new dataset is loaded, the generation increments, causing all pending callbacks from the previous load to exit early.

#### 4. Channel Label Update Logic (`TestChannelLabelUpdate`)

Tests the logic that applies channel names to NDV LUT controllers (in `_update_channel_labels` method):

- **Setting controller keys**: Tests that channel names are correctly assigned to controller.key
- **Fewer controllers than names**: Tests graceful handling when there are more channel names than controllers
- **Missing controller indices**: Tests that missing indices are skipped without errors

#### 5. Integration Tests (`TestChannelNamesIntegration`)

Tests the end-to-end behavior of channel name handling:

- **Length consistency**: Verifies channel names list always matches channel dimension size
- **Default format**: Verifies default names follow `Ch0`, `Ch1`, `Ch2` format
- **Empty/None handling**: Tests that empty lists and None values become defaults
- **Custom name preservation**: Tests that custom names are preserved when counts match

## Test Philosophy

These tests follow a **logic-focused** approach rather than integration testing:

1. **Isolated from Qt**: Tests avoid Qt initialization to prevent crashes and improve speed
2. **Logic verification**: Tests directly verify the channel name processing logic by simulating the same code paths
3. **Edge case coverage**: Comprehensive coverage of boundary conditions (empty, fewer, more, exact match)
4. **Clear documentation**: Each test has a descriptive name and docstring explaining what it verifies

This approach ensures:
- Fast test execution (typically under 1 second for the full suite)
- No dependency on GUI components
- Clear verification of business logic
- Easy debugging when tests fail

## Test Organization

Tests are organized into classes by functional area:
- `TestOMETiffChannelNamesLogic`: OME-TIFF path logic
- `TestSingleTiffChannelNamesLogic`: Single-TIFF path logic
- `TestChannelLabelRetryMechanism`: Retry and generation counter logic
- `TestChannelLabelUpdate`: Controller update logic
- `TestChannelNamesIntegration`: End-to-end integration tests

## Code References

Each test includes comments referencing the relevant methods in `ndviewer_light.py` that implement the tested logic, making it easy to trace tests back to the implementation using IDE navigation (e.g., "Go to Definition").
