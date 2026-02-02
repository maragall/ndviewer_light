"""Tests for Zarr v3 reading support.

Tests verify:
- Format detection for zarr v3 datasets
- Metadata parsing from zarr.json files
- FOV discovery for HCS and non-HCS structures
- Hex color to colormap conversion
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from ndviewer_light import (
    detect_format,
    discover_zarr_v3_fovs,
    hex_to_colormap,
    open_zarr_tensorstore,
    parse_zarr_v3_metadata,
)


def _write_zarr_json(path: Path, attrs: dict = None) -> None:
    """Helper to write a valid zarr.json with optional attributes."""
    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
    }
    if attrs:
        zarr_json["attributes"] = attrs
    (path / "zarr.json").write_text(json.dumps(zarr_json))


class TestDetectZarrV3Format:
    """Test suite for zarr v3 format detection."""

    def test_detect_hcs_plate_zarr(self):
        """Test detection of HCS plate.zarr structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create plate.zarr directory
            plate_zarr = base / "plate.zarr"
            plate_zarr.mkdir()
            _write_zarr_json(plate_zarr)

            assert detect_format(base) == "zarr_v3"

    def test_detect_non_hcs_zarr_directory(self):
        """Test detection of zarr/ directory with .zarr subdirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create zarr/region/acquisition.zarr structure
            zarr_dir = base / "zarr" / "region_1"
            zarr_dir.mkdir(parents=True)
            acq_zarr = zarr_dir / "acquisition.zarr"
            acq_zarr.mkdir()
            _write_zarr_json(acq_zarr)

            assert detect_format(base) == "zarr_v3"

    def test_detect_non_hcs_fov_zarr(self):
        """Test detection of zarr/region/fov_N.zarr structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create zarr/region/fov_0.zarr structure
            zarr_dir = base / "zarr" / "region_1"
            zarr_dir.mkdir(parents=True)
            fov_zarr = zarr_dir / "fov_0.zarr"
            fov_zarr.mkdir()
            _write_zarr_json(fov_zarr)

            assert detect_format(base) == "zarr_v3"

    def test_detect_direct_zarr_directory(self):
        """Test detection when base_path is a .zarr directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "dataset.zarr"
            base.mkdir()
            _write_zarr_json(base)

            assert detect_format(base) == "zarr_v3"

    def test_detect_direct_zarr_with_zarr_json(self):
        """Test detection with zarr.json file (zarr v3 format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "dataset.zarr"
            base.mkdir()
            (base / "zarr.json").write_text('{"zarr_format": 3}')

            assert detect_format(base) == "zarr_v3"

    def test_detect_ome_tiff_not_zarr(self):
        """Test that OME-TIFF is not detected as zarr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ome_dir = base / "ome_tiff"
            ome_dir.mkdir()
            (ome_dir / "test_0.ome.tiff").touch()

            assert detect_format(base) == "ome_tiff"

    def test_detect_single_tiff_fallback(self):
        """Test fallback to single_tiff when no zarr or ome_tiff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            tp_dir = base / "0"
            tp_dir.mkdir()
            (tp_dir / "region_0_0_channel.tiff").touch()

            assert detect_format(base) == "single_tiff"


class TestParseZarrV3Metadata:
    """Test suite for zarr v3 metadata parsing."""

    def test_parse_ome_ngff_multiscales(self):
        """Test parsing OME-NGFF multiscales metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            attrs = {
                "ome": {
                    "multiscales": [
                        {
                            "axes": [
                                {"name": "t", "type": "time"},
                                {"name": "c", "type": "channel"},
                                {"name": "z", "type": "space", "unit": "micrometer"},
                                {"name": "y", "type": "space", "unit": "micrometer"},
                                {"name": "x", "type": "space", "unit": "micrometer"},
                            ],
                            "datasets": [
                                {
                                    "path": "0",
                                    "coordinateTransformations": [
                                        {
                                            "type": "scale",
                                            "scale": [1, 1, 2.0, 0.325, 0.325],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            }
            _write_zarr_json(zarr_path, attrs)

            meta = parse_zarr_v3_metadata(zarr_path)

            assert len(meta["axes"]) == 5
            assert meta["pixel_size_um"] == pytest.approx(0.325)
            assert meta["dz_um"] == pytest.approx(2.0)

    def test_parse_omero_channels(self):
        """Test parsing omero.channels metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            attrs = {
                "ome": {
                    "omero": {
                        "channels": [
                            {"label": "DAPI", "color": "0000FF"},
                            {"label": "GFP", "color": "00FF00"},
                            {"label": "RFP", "color": "FF0000"},
                        ]
                    }
                }
            }
            _write_zarr_json(zarr_path, attrs)

            meta = parse_zarr_v3_metadata(zarr_path)

            assert meta["channel_names"] == ["DAPI", "GFP", "RFP"]
            assert meta["channel_colors"] == ["0000FF", "00FF00", "FF0000"]

    def test_parse_squid_metadata(self):
        """Test parsing _squid metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            attrs = {
                "_squid": {
                    "pixel_size_um": 0.5,
                    "z_step_um": 1.5,
                    "acquisition_complete": True,
                }
            }
            _write_zarr_json(zarr_path, attrs)

            meta = parse_zarr_v3_metadata(zarr_path)

            assert meta["pixel_size_um"] == pytest.approx(0.5)
            assert meta["dz_um"] == pytest.approx(1.5)
            assert meta["acquisition_complete"] is True

    def test_squid_metadata_overrides_multiscales(self):
        """Test that _squid metadata overrides multiscales values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            attrs = {
                "ome": {
                    "multiscales": [
                        {
                            "axes": [
                                {"name": "z", "type": "space", "unit": "micrometer"},
                                {"name": "y", "type": "space", "unit": "micrometer"},
                                {"name": "x", "type": "space", "unit": "micrometer"},
                            ],
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 0.325, 0.325]}
                            ],
                        }
                    ]
                },
                "_squid": {
                    "pixel_size_um": 0.5,
                    "z_step_um": 2.0,
                },
            }
            _write_zarr_json(zarr_path, attrs)

            meta = parse_zarr_v3_metadata(zarr_path)

            # Should use _squid values
            assert meta["pixel_size_um"] == pytest.approx(0.5)
            assert meta["dz_um"] == pytest.approx(2.0)

    def test_parse_missing_zarr_json(self):
        """Test handling of missing zarr.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)

            meta = parse_zarr_v3_metadata(zarr_path)

            assert meta["axes"] == []
            assert meta["pixel_size_um"] is None
            assert meta["dz_um"] is None
            assert meta["channel_names"] == []
            assert meta["acquisition_complete"] is False

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON in zarr.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            (zarr_path / "zarr.json").write_text("not valid json {")

            meta = parse_zarr_v3_metadata(zarr_path)

            assert meta["axes"] == []
            assert meta["pixel_size_um"] is None


class TestDiscoverZarrV3Fovs:
    """Test suite for zarr v3 FOV discovery."""

    def test_discover_hcs_plate_fovs(self):
        """Test FOV discovery in HCS plate structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create plate.zarr/A/1/0/acquisition.zarr
            for row in ["A", "B"]:
                for col in ["1", "2"]:
                    for field in ["0", "1"]:
                        acq_path = (
                            base / "plate.zarr" / row / col / field / "acquisition.zarr"
                        )
                        acq_path.mkdir(parents=True)
                        _write_zarr_json(acq_path)

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "hcs_plate"
            assert len(fovs) == 8  # 2 rows * 2 cols * 2 fields
            # Check first FOV
            assert fovs[0]["region"] == "A1"
            assert fovs[0]["fov"] == 0
            assert fovs[0]["path"].exists()

    def test_discover_non_hcs_per_fov(self):
        """Test FOV discovery in per-FOV structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create zarr/region_1/fov_0.zarr, fov_1.zarr
            region_dir = base / "zarr" / "region_1"
            region_dir.mkdir(parents=True)
            for fov_idx in range(3):
                fov_zarr = region_dir / f"fov_{fov_idx}.zarr"
                fov_zarr.mkdir()
                _write_zarr_json(fov_zarr)

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "per_fov"
            assert len(fovs) == 3
            assert fovs[0]["region"] == "region_1"
            assert fovs[0]["fov"] == 0

    def test_discover_6d(self):
        """Test FOV discovery for 6D single store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create zarr/region_1/acquisition.zarr
            acq_path = base / "zarr" / "region_1" / "acquisition.zarr"
            acq_path.mkdir(parents=True)
            _write_zarr_json(acq_path)

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "6d"
            assert len(fovs) == 1
            assert fovs[0]["region"] == "region_1"

    def test_discover_direct_zarr(self):
        """Test FOV discovery when base_path is a .zarr directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "dataset.zarr"
            base.mkdir()
            _write_zarr_json(base)

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "6d"
            assert len(fovs) == 1
            assert fovs[0]["path"] == base

    def test_discover_empty_returns_unknown(self):
        """Test that empty directory returns unknown structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "unknown"
            assert len(fovs) == 0


class TestHexToColormap:
    """Test suite for hex color to colormap conversion."""

    def test_blue_colors(self):
        """Test blue hex colors map to blue colormap."""
        assert hex_to_colormap("#0000FF") == "blue"
        assert hex_to_colormap("0000FF") == "blue"
        assert hex_to_colormap("#0033CC") == "blue"

    def test_green_colors(self):
        """Test green hex colors map to green colormap."""
        assert hex_to_colormap("#00FF00") == "green"
        assert hex_to_colormap("#00CC00") == "green"

    def test_red_colors(self):
        """Test red hex colors map to red colormap."""
        assert hex_to_colormap("#FF0000") == "red"
        assert hex_to_colormap("#CC0000") == "red"

    def test_yellow_colors(self):
        """Test yellow hex colors map to yellow colormap."""
        assert hex_to_colormap("#FFFF00") == "yellow"
        assert hex_to_colormap("#CCCC00") == "yellow"

    def test_magenta_colors(self):
        """Test magenta hex colors map to magenta colormap."""
        assert hex_to_colormap("#FF00FF") == "magenta"
        assert hex_to_colormap("#CC00CC") == "magenta"

    def test_cyan_colors(self):
        """Test cyan hex colors map to cyan colormap."""
        assert hex_to_colormap("#00FFFF") == "cyan"
        assert hex_to_colormap("#00CCCC") == "cyan"

    def test_gray_colors(self):
        """Test gray hex colors map to gray colormap."""
        assert hex_to_colormap("#808080") == "gray"
        assert hex_to_colormap("#A0A0A0") == "gray"

    def test_squid_colors(self):
        """Test specific colors used by Squid."""
        # DAPI blue
        assert hex_to_colormap("#20ADF8") == "cyan"  # Nearest to cyan
        # GFP green
        assert hex_to_colormap("#00FF00") == "green"
        # RFP red
        assert hex_to_colormap("#FF0000") == "red"

    def test_empty_color(self):
        """Test empty color returns gray."""
        assert hex_to_colormap("") == "gray"
        assert hex_to_colormap(None) == "gray"

    def test_invalid_hex(self):
        """Test invalid hex returns gray."""
        assert hex_to_colormap("not-hex") == "gray"
        assert hex_to_colormap("#12") == "gray"  # Too short
        assert hex_to_colormap("#GGGGGG") == "gray"  # Invalid chars


class TestNewSquidFormat:
    """Test suite for new Squid zarr format (PR #474)."""

    def test_detect_plate_ome_zarr(self):
        """Test detection of plate.ome.zarr (new HCS format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            plate_zarr = base / "plate.ome.zarr"
            plate_zarr.mkdir()
            (plate_zarr / "zarr.json").write_text('{"zarr_format": 3}')

            assert detect_format(base) == "zarr_v3"

    def test_detect_fov_ome_zarr(self):
        """Test detection of fov_N.ome.zarr (new per-FOV format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            zarr_dir = base / "zarr" / "region_1"
            zarr_dir.mkdir(parents=True)
            fov_zarr = zarr_dir / "fov_0.ome.zarr"
            fov_zarr.mkdir()
            (fov_zarr / "zarr.json").write_text('{"zarr_format": 3}')

            assert detect_format(base) == "zarr_v3"

    def test_parse_metadata_from_zarr_json(self):
        """Test parsing metadata from zarr.json attributes (new format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            zarr_json = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "multiscales": [
                            {
                                "axes": [
                                    {"name": "t", "type": "time"},
                                    {"name": "c", "type": "channel"},
                                    {
                                        "name": "z",
                                        "type": "space",
                                        "unit": "micrometer",
                                    },
                                    {
                                        "name": "y",
                                        "type": "space",
                                        "unit": "micrometer",
                                    },
                                    {
                                        "name": "x",
                                        "type": "space",
                                        "unit": "micrometer",
                                    },
                                ],
                                "datasets": [
                                    {
                                        "path": "0",
                                        "coordinateTransformations": [
                                            {
                                                "type": "scale",
                                                "scale": [1, 1, 2.0, 0.325, 0.325],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                        "omero": {
                            "channels": [
                                {"label": "DAPI", "color": "0000FF"},
                                {"label": "GFP", "color": "00FF00"},
                            ]
                        },
                    },
                    "_squid": {
                        "pixel_size_um": 0.5,
                        "z_step_um": 1.5,
                        "acquisition_complete": True,
                    },
                },
            }
            (zarr_path / "zarr.json").write_text(json.dumps(zarr_json))

            meta = parse_zarr_v3_metadata(zarr_path)

            assert meta["channel_names"] == ["DAPI", "GFP"]
            assert meta["channel_colors"] == ["0000FF", "00FF00"]
            assert meta["pixel_size_um"] == pytest.approx(0.5)
            assert meta["dz_um"] == pytest.approx(1.5)
            assert meta["acquisition_complete"] is True

    def test_discover_new_hcs_structure(self):
        """Test FOV discovery for new HCS plate.ome.zarr/{row}/{col}/{fov}/0 structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Create plate.ome.zarr/A/1/0/0 (row/col/fov/array)
            for row in ["A", "B"]:
                for col in ["1", "2"]:
                    for fov in ["0", "1"]:
                        fov_path = base / "plate.ome.zarr" / row / col / fov
                        fov_path.mkdir(parents=True)
                        # Create "0" array directory (new format indicator)
                        (fov_path / "0").mkdir()
                        (fov_path / "zarr.json").write_text('{"zarr_format": 3}')

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "hcs_plate"
            assert len(fovs) == 8  # 2 rows * 2 cols * 2 fovs
            assert fovs[0]["region"] == "A1"
            assert fovs[0]["fov"] == 0

    def test_discover_new_per_fov_structure(self):
        """Test FOV discovery for new zarr/region/fov_N.ome.zarr structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            region_dir = base / "zarr" / "region_1"
            region_dir.mkdir(parents=True)
            for fov_idx in range(3):
                fov_zarr = region_dir / f"fov_{fov_idx}.ome.zarr"
                fov_zarr.mkdir()
                (fov_zarr / "zarr.json").write_text('{"zarr_format": 3}')

            fovs, structure_type = discover_zarr_v3_fovs(base)

            assert structure_type == "per_fov"
            assert len(fovs) == 3
            assert fovs[0]["region"] == "region_1"
            assert fovs[0]["fov"] == 0

    def test_zarr_json_without_attributes(self):
        """Test that zarr.json without attributes returns default metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir)
            # zarr.json without attributes
            (zarr_path / "zarr.json").write_text('{"zarr_format": 3}')

            meta = parse_zarr_v3_metadata(zarr_path)

            # Should return defaults when no attributes present
            assert meta["axes"] == []
            assert meta["pixel_size_um"] is None
            assert meta["channel_names"] == []


class TestOpenZarrTensorstore:
    """Test suite for tensorstore-based zarr loading."""

    def _create_zarr_v3_array(
        self, path: Path, shape: tuple, dtype=np.uint16
    ) -> np.ndarray:
        """Helper to create a zarr v3 array with tensorstore."""
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(shape),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(shape)},
                },
                "chunk_key_encoding": {"name": "default"},
                "data_type": dtype.__name__,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        # Write test data
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        store[...] = data
        return data

    def test_open_zarr_v3_array(self):
        """Test opening a zarr v3 array with tensorstore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"
            zarr_path.mkdir()
            _write_zarr_json(zarr_path)

            # Create array at "0" path (OME-NGFF convention)
            array_path = zarr_path / "0"
            shape = (2, 3, 64, 64)  # T, C, Y, X
            expected_data = self._create_zarr_v3_array(array_path, shape)

            # Open with our function
            arr = open_zarr_tensorstore(zarr_path, array_path="0")

            assert arr is not None
            assert arr.shape == shape
            assert arr.dtype == np.uint16
            # Verify data matches
            np.testing.assert_array_equal(arr[...].read().result(), expected_data)

    def test_open_zarr_v3_5d_shape(self):
        """Test opening a 5D zarr array (T, C, Z, Y, X)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "fov.zarr"
            zarr_path.mkdir()
            _write_zarr_json(zarr_path)

            array_path = zarr_path / "0"
            shape = (3, 2, 5, 128, 128)  # T, C, Z, Y, X
            self._create_zarr_v3_array(array_path, shape)

            arr = open_zarr_tensorstore(zarr_path, array_path="0")

            assert arr is not None
            assert arr.shape == shape
            assert len(arr.shape) == 5

    def test_open_zarr_v3_6d_shape(self):
        """Test opening a 6D zarr array (FOV, T, C, Z, Y, X)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "acquisition.zarr"
            zarr_path.mkdir()
            _write_zarr_json(zarr_path)

            array_path = zarr_path / "0"
            shape = (4, 2, 3, 5, 64, 64)  # FOV, T, C, Z, Y, X
            self._create_zarr_v3_array(array_path, shape)

            arr = open_zarr_tensorstore(zarr_path, array_path="0")

            assert arr is not None
            assert arr.shape == shape
            assert len(arr.shape) == 6

    def test_open_nonexistent_path_returns_none(self):
        """Test that opening a nonexistent path returns None."""
        arr = open_zarr_tensorstore(Path("/nonexistent/path.zarr"), array_path="0")
        assert arr is None

    def test_open_invalid_zarr_returns_none(self):
        """Test that opening an invalid zarr store returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "invalid.zarr"
            zarr_path.mkdir()
            # Write invalid zarr.json
            (zarr_path / "zarr.json").write_text("not valid json")

            arr = open_zarr_tensorstore(zarr_path, array_path="0")
            assert arr is None

    def test_tensorstore_slicing(self):
        """Test that tensorstore array supports slicing for lazy loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"
            zarr_path.mkdir()
            _write_zarr_json(zarr_path)

            array_path = zarr_path / "0"
            shape = (2, 3, 4, 32, 32)  # T, C, Z, Y, X
            expected_data = self._create_zarr_v3_array(array_path, shape)

            arr = open_zarr_tensorstore(zarr_path, array_path="0")

            # Test single plane extraction (what viewer does)
            plane = arr[0, 1, 2, :, :].read().result()
            assert plane.shape == (32, 32)
            np.testing.assert_array_equal(plane, expected_data[0, 1, 2, :, :])

            # Test range slicing
            subset = arr[0:1, 0:2, :, :, :].read().result()
            assert subset.shape == (1, 2, 4, 32, 32)
