"""
Unit tests for get_fov_list() method in LightweightViewer.

Tests use mocks to avoid Qt dependencies while testing the actual method.
"""

from unittest.mock import MagicMock


class TestGetFovList:
    """Tests for LightweightViewer.get_fov_list() method."""

    def _create_mock_viewer(self, dataset_path=None):
        """Create a mock viewer with get_fov_list method from real class."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        mock.dataset_path = dataset_path
        # Bind the real methods to our mock
        mock.get_fov_list = lambda: LightweightViewer.get_fov_list(mock)
        mock._discover_fovs = LightweightViewer._discover_fovs.__get__(mock)
        return mock

    def test_returns_empty_list_when_no_dataset_path(self):
        """When dataset_path is not set, should return empty list."""
        viewer = self._create_mock_viewer(dataset_path=None)
        result = viewer.get_fov_list()
        assert result == []

    def test_returns_empty_list_when_dataset_path_empty(self):
        """When dataset_path is empty string, should return empty list."""
        viewer = self._create_mock_viewer(dataset_path="")
        result = viewer.get_fov_list()
        assert result == []

    def test_handles_nonexistent_path_gracefully(self):
        """When dataset_path doesn't exist, should return empty list."""
        viewer = self._create_mock_viewer(dataset_path="/nonexistent/path/to/dataset")
        result = viewer.get_fov_list()
        assert result == []


class TestGetFovListIntegration:
    """Integration tests for get_fov_list() with real filesystem."""

    def _create_mock_viewer_with_discover(self, dataset_path):
        """Create mock viewer with real get_fov_list and _discover_fovs."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        mock.dataset_path = dataset_path
        mock.get_fov_list = lambda: LightweightViewer.get_fov_list(mock)
        mock._discover_fovs = LightweightViewer._discover_fovs.__get__(mock)
        return mock

    def test_ome_tiff_discovery(self, tmp_path):
        """Test FOV discovery for OME-TIFF format datasets."""
        # Create correct directory structure: dataset_root/ome_tiff/*.ome.tif
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        ome_dir = dataset_root / "ome_tiff"
        ome_dir.mkdir()

        # Create test OME-TIFF files
        test_files = [
            "A1_0.ome.tif",
            "A1_1.ome.tif",
            "A2_0.ome.tif",
            "B1_0.ome.tif",
        ]
        for fname in test_files:
            (ome_dir / fname).touch()

        viewer = self._create_mock_viewer_with_discover(str(dataset_root))
        result = viewer.get_fov_list()

        # Verify results are sorted by region then FOV
        assert len(result) == 4
        assert result[0] == {"region": "A1", "fov": 0}
        assert result[1] == {"region": "A1", "fov": 1}
        assert result[2] == {"region": "A2", "fov": 0}
        assert result[3] == {"region": "B1", "fov": 0}

    def test_single_tiff_discovery(self, tmp_path):
        """Test FOV discovery for single-TIFF format datasets."""
        # Create correct structure: dataset_root/0/*.tiff (timepoint dir)
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        timepoint_dir = dataset_root / "0"
        timepoint_dir.mkdir()

        # Create test TIFF files: <region>_<fov>_<z>_<channel>.tiff
        test_files = [
            "A1_0_0_405.tiff",
            "A1_1_0_405.tiff",
            "A1_0_0_488.tiff",  # Same FOV, different channel
            "B1_0_0_405.tiff",
        ]
        for fname in test_files:
            (timepoint_dir / fname).touch()

        viewer = self._create_mock_viewer_with_discover(str(dataset_root))
        result = viewer.get_fov_list()

        # Verify unique FOVs, sorted
        assert len(result) == 3  # A1_0, A1_1, B1_0
        assert result[0] == {"region": "A1", "fov": 0}
        assert result[1] == {"region": "A1", "fov": 1}
        assert result[2] == {"region": "B1", "fov": 0}

    def test_fov_list_sorted_by_region_then_fov(self, tmp_path):
        """Verify FOV list is sorted by region first, then by FOV index."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        ome_dir = dataset_root / "ome_tiff"
        ome_dir.mkdir()

        # Create files in non-sorted order
        test_files = [
            "B2_1.ome.tif",
            "A1_0.ome.tif",
            "B1_0.ome.tif",
            "A2_1.ome.tif",
            "A1_2.ome.tif",
        ]
        for fname in test_files:
            (ome_dir / fname).touch()

        viewer = self._create_mock_viewer_with_discover(str(dataset_root))
        result = viewer.get_fov_list()

        expected = [
            {"region": "A1", "fov": 0},
            {"region": "A1", "fov": 2},
            {"region": "A2", "fov": 1},
            {"region": "B1", "fov": 0},
            {"region": "B2", "fov": 1},
        ]
        assert result == expected

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """When dataset directory has no matching files."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        ome_dir = dataset_root / "ome_tiff"
        ome_dir.mkdir()
        # No files created

        viewer = self._create_mock_viewer_with_discover(str(dataset_root))
        result = viewer.get_fov_list()

        assert result == []

    def test_non_matching_filenames_ignored(self, tmp_path):
        """Files that don't match the pattern should be ignored."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        ome_dir = dataset_root / "ome_tiff"
        ome_dir.mkdir()

        # Create files that don't match the pattern
        non_matching = [
            "random_file.ome.tif",
            "no_fov.ome.tif",
            "image.tif",  # Not .ome.tif
        ]
        for fname in non_matching:
            (ome_dir / fname).touch()

        # Create one valid file
        (ome_dir / "A1_0.ome.tif").touch()

        viewer = self._create_mock_viewer_with_discover(str(dataset_root))
        result = viewer.get_fov_list()

        # Only the valid file should be discovered
        assert len(result) == 1
        assert result[0] == {"region": "A1", "fov": 0}
