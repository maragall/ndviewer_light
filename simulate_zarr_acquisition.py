"""
Simulate a live acquisition using the push-based zarr API.

This script tests the zarr v3 push-based API with different zarr structures:
- per_fov: Separate zarr per FOV: zarr/region/fov_N.zarr (5D: T, C, Z, Y, X)
- 6d: 6D with FOV dimension (FOV, T, C, Z, Y, X), supports multi-region
- hcs: HCS plate format: plate.zarr/row/col/field/acquisition.zarr

Usage:
    python simulate_zarr_acquisition.py --structure per_fov --n-fov 4
    python simulate_zarr_acquisition.py --structure per_fov --n-fov 1  # Single FOV
    python simulate_zarr_acquisition.py --structure 6d --n-fov 4
    python simulate_zarr_acquisition.py --structure 6d --n-regions 3 --fovs-per-region 4 6 3
    python simulate_zarr_acquisition.py --structure hcs --wells A1 A2 B1 B2 --fov-per-well 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorstore as ts
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from ndviewer_light import LightweightViewer

_FONT_5X7: dict[str, list[str]] = {
    "0": ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": ["#####", "....#", "....#", "#####", "#....", "#....", "#####"],
    "3": ["#####", "....#", "....#", "#####", "....#", "....#", "#####"],
    "4": ["#...#", "#...#", "#...#", "#####", "....#", "....#", "....#"],
    "5": ["#####", "#....", "#....", "#####", "....#", "....#", "#####"],
    "6": ["#####", "#....", "#....", "#####", "#...#", "#...#", "#####"],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": ["#####", "#...#", "#...#", "#####", "#...#", "#...#", "#####"],
    "9": ["#####", "#...#", "#...#", "#####", "....#", "....#", "#####"],
    "F": ["#####", "#....", "#....", "#####", "#....", "#....", "#...."],
    "O": ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "C": ["#####", "#....", "#....", "#....", "#....", "#....", "#####"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "R": ["####.", "#...#", "#...#", "####.", "#..#.", "#...#", "#...#"],
    "=": [".....", "#####", ".....", "#####", ".....", ".....", "....."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],
    ":": [".....", "..#..", ".....", ".....", "..#..", ".....", "....."],
}


def _draw_text(
    img: np.ndarray, text: str, x: int, y: int, scale: int, value: int
) -> None:
    """Draw text into a uint16 image in-place using the bitmap font."""
    h, w = img.shape
    cursor_x = x
    cursor_y = y
    char_w = 5 * scale
    spacing = 1 * scale

    for ch in text:
        glyph = _FONT_5X7.get(ch.upper())
        if glyph is None:
            glyph = _FONT_5X7[" "]

        if cursor_x >= w or cursor_y >= h:
            break

        for gy in range(7):
            row = glyph[gy]
            for gx in range(5):
                if row[gx] != "#":
                    continue
                px0 = cursor_x + gx * scale
                py0 = cursor_y + gy * scale
                px1 = min(w, px0 + scale)
                py1 = min(h, py0 + scale)
                if px0 < 0 or py0 < 0 or px0 >= w or py0 >= h:
                    continue
                img[py0:py1, px0:px1] = np.uint16(value)

        cursor_x += char_w + spacing


def _write_zarr_metadata(
    zarr_path: Path,
    channels: list[str],
    channel_colors: list[str],
    pixel_size_um: float,
    z_step_um: float,
    acquisition_complete: bool = False,
    axes: Optional[list[dict]] = None,
    scale: Optional[list[float]] = None,
    merge_into_existing: bool = False,
) -> None:
    """Write OME-NGFF metadata to zarr.json (zarr v3 format).

    Writes metadata to zarr.json -> attributes with ome.multiscales/omero and _squid.

    Args:
        merge_into_existing: If True, merge attributes into existing zarr.json
            (for 6D mode where array is at root). If False, create new group zarr.json.
    """
    if axes is None:
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
    if scale is None:
        scale = [1, 1, z_step_um, pixel_size_um, pixel_size_um]

    # Build OME-NGFF metadata
    attributes = {
        "ome": {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": axes,
                    "datasets": [
                        {
                            # "." when array is at root (6D), "0" when at /0 subdirectory
                            "path": "." if merge_into_existing else "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": scale}
                            ],
                        }
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {"label": name, "color": color}
                    for name, color in zip(channels, channel_colors)
                ]
            },
        },
        "_squid": {
            "pixel_size_um": pixel_size_um,
            "z_step_um": z_step_um,
            "acquisition_complete": acquisition_complete,
        },
    }

    zarr_json_path = zarr_path / "zarr.json"

    if merge_into_existing and zarr_json_path.exists():
        # Merge attributes into existing array zarr.json (for 6D mode)
        with open(zarr_json_path, "r") as f:
            zarr_json = json.load(f)
        zarr_json["attributes"] = attributes
    else:
        # Create new group zarr.json (for per_fov/hcs modes)
        zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attributes,
        }

    with open(zarr_json_path, "w") as f:
        json.dump(zarr_json, f, indent=2)


class ZarrAcquisitionSimulator:
    """Simulates acquisition by writing to zarr and calling notify_zarr_frame()."""

    def __init__(
        self,
        viewer: LightweightViewer,
        output_path: Path,
        structure: str,
        n_fov: int,
        n_z: int,
        n_t: int,
        channels: list[str],
        channel_colors: list[str],
        height: int,
        width: int,
        interval_ms: int,
        pixel_size_um: float = 0.325,
        z_step_um: float = 1.5,
        wells: Optional[list[str]] = None,
        fov_per_well: int = 1,
        n_regions: int = 1,
        fovs_per_region: Optional[list[int]] = None,
    ):
        self.viewer = viewer
        self.output_path = output_path
        self.structure = structure
        self.n_fov = n_fov
        self.n_z = n_z
        self.n_t = n_t
        self.channels = channels
        self.channel_colors = channel_colors
        self.height = height
        self.width = width
        self.interval_ms = interval_ms
        self.pixel_size_um = pixel_size_um
        self.z_step_um = z_step_um
        self.wells = wells or ["A1"]
        self.fov_per_well = fov_per_well
        self.n_regions = n_regions
        self.fovs_per_region = fovs_per_region or [n_fov]

        # Current position in acquisition
        self.current_t = 0
        self.current_fov = 0  # Global FOV index
        self.current_z = 0
        self.current_c = 0
        self.current_region = 0  # For 6d_regions: current region index

        # Precompute base image pattern
        y = np.arange(height, dtype=np.uint16)[:, None]
        x = np.arange(width, dtype=np.uint16)[None, :]
        self.base = y + x

        # Generate FOV labels based on structure
        self.fov_labels = []
        self.fov_paths = []  # For per_fov and hcs structures
        self._setup_fov_structure()

        # Timer for periodic writes (one plane at a time)
        self.timer = QTimer()
        self.timer.timeout.connect(self._write_next_plane)

        # Tensorstore array handles (fov_idx or region_idx -> ts.TensorStore)
        self.zarr_arrays = {}

    def _setup_fov_structure(self):
        """Set up FOV labels and paths based on structure type."""
        if self.structure == "6d":
            # 6D: each region has its own zarr with (FOV, T, C, Z, Y, X)
            # Supports single region (n_regions=1) and multi-region
            # Compute cumulative offsets for global→local FOV conversion
            self.region_fov_offsets = []
            offset = 0
            for n_fov in self.fovs_per_region:
                self.region_fov_offsets.append(offset)
                offset += n_fov

            # Generate flattened FOV labels and region paths
            self.region_labels = [f"region_{i}" for i in range(self.n_regions)]
            for region_idx, (region_label, n_fov) in enumerate(
                zip(self.region_labels, self.fovs_per_region)
            ):
                for fov_in_region in range(n_fov):
                    self.fov_labels.append(f"{region_label}:{fov_in_region}")
                region_path = (
                    self.output_path / "zarr" / region_label / "acquisition.zarr"
                )
                self.fov_paths.append(region_path)

            self.n_fov = sum(self.fovs_per_region)

        elif self.structure == "hcs":
            # HCS plate: plate.zarr/row/col/field/acquisition.zarr
            for well in self.wells:
                row = well[0]  # e.g., "A"
                col = well[1:]  # e.g., "1"
                for field in range(self.fov_per_well):
                    self.fov_labels.append(f"{well}:{field}")
                    acq_path = (
                        self.output_path
                        / "plate.zarr"
                        / row
                        / col
                        / str(field)
                        / "acquisition.zarr"
                    )
                    self.fov_paths.append(acq_path)
            self.n_fov = len(self.fov_labels)

        elif self.structure == "per_fov":
            # Per-FOV: zarr/region_1/fov_N.zarr
            for i in range(self.n_fov):
                well_idx = i // max(1, self.n_fov // len(self.wells))
                well = self.wells[well_idx % len(self.wells)]
                fov_in_well = i % max(1, self.n_fov // len(self.wells))
                self.fov_labels.append(f"{well}:{fov_in_well}")
                fov_path = self.output_path / "zarr" / "region_1" / f"fov_{i}.zarr"
                self.fov_paths.append(fov_path)

        else:  # single
            # Simple 5D single store
            self.fov_labels = ["A1:0"]
            self.n_fov = 1
            self.fov_paths = [self.output_path]
            if not str(self.output_path).endswith(".zarr"):
                self.fov_paths = [self.output_path.with_suffix(".zarr")]

    def _create_tensorstore_array(
        self, path: Path, shape: tuple, chunks: tuple
    ) -> ts.TensorStore:
        """Create a zarr v3 array using tensorstore."""
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "create": True,
            "delete_existing": True,
            "schema": {
                "dtype": "uint16",
                "domain": {"shape": list(shape)},
                "chunk_layout": {
                    "grid_origin": [0] * len(shape),
                    "inner_order": list(range(len(shape) - 1, -1, -1)),
                    "chunk": {"shape": list(chunks)},
                },
            },
        }
        return ts.open(spec).result()

    def _create_zarr_stores(self):
        """Create zarr v3 stores using tensorstore based on structure type."""
        n_c = len(self.channels)

        if self.structure == "6d":
            # 6D: each region has its own store with (FOV, T, C, Z, Y, X)
            for region_idx, (region_label, n_fov_in_region) in enumerate(
                zip(self.region_labels, self.fovs_per_region)
            ):
                zarr_path = self.fov_paths[region_idx]  # fov_paths holds region paths
                zarr_path.mkdir(parents=True, exist_ok=True)
                # Shape: (FOV, T, C, Z, Y, X)
                shape = (
                    n_fov_in_region,
                    self.n_t,
                    n_c,
                    self.n_z,
                    self.height,
                    self.width,
                )
                chunks = (1, 1, 1, 1, self.height, self.width)
                # 6D mode: array directly at acquisition.zarr root (not /0 subdirectory)
                arr = self._create_tensorstore_array(zarr_path, shape, chunks)
                self.zarr_arrays[region_idx] = arr
                _write_zarr_metadata(
                    zarr_path,
                    self.channels,
                    self.channel_colors,
                    self.pixel_size_um,
                    self.z_step_um,
                    axes=[
                        {"name": "fov", "type": "position"},
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    scale=[
                        1,
                        1,
                        1,
                        self.z_step_um,
                        self.pixel_size_um,
                        self.pixel_size_um,
                    ],
                    merge_into_existing=True,  # 6D: array is at root, merge attrs
                )
            print(f"Created 6D zarr v3 stores: {self.n_regions} regions")
            for i, (label, n_fov) in enumerate(
                zip(self.region_labels, self.fovs_per_region)
            ):
                print(f"  [{i}] {label}: {n_fov} FOVs at {self.fov_paths[i]}")

        elif self.structure in ("per_fov", "hcs"):
            # Separate store per FOV: (T, C, Z, Y, X) each
            for fov_idx, zarr_path in enumerate(self.fov_paths):
                zarr_path.mkdir(parents=True, exist_ok=True)
                shape = (self.n_t, n_c, self.n_z, self.height, self.width)
                chunks = (1, 1, 1, self.height, self.width)
                arr = self._create_tensorstore_array(zarr_path / "0", shape, chunks)
                self.zarr_arrays[fov_idx] = arr
                _write_zarr_metadata(
                    zarr_path,
                    self.channels,
                    self.channel_colors,
                    self.pixel_size_um,
                    self.z_step_um,
                )
            struct_name = "HCS plate" if self.structure == "hcs" else "per-FOV"
            print(f"Created {struct_name} zarr v3 stores: {len(self.fov_paths)} FOVs")
            for i, p in enumerate(self.fov_paths[:3]):
                print(f"  [{i}] {p}")
            if len(self.fov_paths) > 3:
                print(f"  ... and {len(self.fov_paths) - 3} more")

    def start(self):
        """Start the simulated acquisition."""
        print(f"Starting zarr push-based acquisition simulation")
        print(f"  Structure: {self.structure}")
        print(f"  Output: {self.output_path}")
        print(
            f"  FOVs: {self.n_fov}, Z: {self.n_z}, T: {self.n_t}, Channels: {self.channels}"
        )

        # Create zarr stores
        self._create_zarr_stores()

        # Determine zarr path(s) for viewer
        if self.structure == "6d":
            # 6D mode (single or multi-region)
            self.viewer.start_zarr_acquisition_6d(
                region_paths=[str(p) for p in self.fov_paths],  # Region paths
                channels=self.channels,
                num_z=self.n_z,
                fovs_per_region=self.fovs_per_region,
                height=self.height,
                width=self.width,
                region_labels=self.region_labels,
            )
        else:
            # Per-FOV mode (per_fov, hcs): each FOV has its own 5D zarr
            self.viewer.start_zarr_acquisition(
                fov_paths=[str(p) for p in self.fov_paths],
                channels=self.channels,
                num_z=self.n_z,
                fov_labels=self.fov_labels,
                height=self.height,
                width=self.width,
            )

        # Start writing
        self.timer.start(self.interval_ms)

    def _global_to_region_fov(self, global_fov_idx: int) -> tuple[int, int]:
        """Convert global FOV index to (region_idx, local_fov_idx)."""
        for region_idx, offset in enumerate(self.region_fov_offsets):
            next_offset = (
                self.region_fov_offsets[region_idx + 1]
                if region_idx + 1 < len(self.region_fov_offsets)
                else self.n_fov
            )
            if offset <= global_fov_idx < next_offset:
                return region_idx, global_fov_idx - offset
        return 0, global_fov_idx  # fallback

    def _write_next_plane(self):
        """Write a single plane, then advance to next position."""
        if self.current_t >= self.n_t:
            self._finish()
            return

        t = self.current_t
        fov = self.current_fov  # Global FOV index
        z = self.current_z
        c = self.current_c
        ch_name = self.channels[c]
        fov_label = self.fov_labels[fov]

        # Create image with identifying pattern
        offset = np.uint16(t * 97 + fov * 11 + c * 301 + z * 50)
        img = (self.base + offset).astype(np.uint16, copy=True)

        # Overlay text label
        label = f"T={t:02d} F={fov} Z={z:02d} C={c}"
        _draw_text(img, label, x=20, y=20, scale=10, value=60000)

        # Write to appropriate zarr array based on structure
        if self.structure == "6d":
            # 6D: convert global FOV to region + local FOV
            region_idx, local_fov_idx = self._global_to_region_fov(fov)
            # (FOV, T, C, Z, Y, X)
            self.zarr_arrays[region_idx][local_fov_idx, t, c, z, :, :] = img
            # Notify with region_idx
            self.viewer.notify_zarr_frame(
                t=t,
                fov_idx=local_fov_idx,
                z=z,
                channel=ch_name,
                region_idx=region_idx,
            )
            print(
                f"[t={t}] Region {region_idx} FOV {local_fov_idx} ({fov_label}) z={z} ch={ch_name}"
            )
        else:
            # per_fov or hcs: each FOV has its own 5D array (T, C, Z, Y, X)
            self.zarr_arrays[fov][t, c, z, :, :] = img
            self.viewer.notify_zarr_frame(t=t, fov_idx=fov, z=z, channel=ch_name)
            print(f"[t={t}] FOV {fov} ({fov_label}) z={z} ch={ch_name}")

        # Advance to next plane: cycle through channels, then z, then FOV, then time
        self.current_c += 1
        if self.current_c >= len(self.channels):
            self.current_c = 0
            self.current_z += 1
            if self.current_z >= self.n_z:
                self.current_z = 0
                self.current_fov += 1
                if self.current_fov >= self.n_fov:
                    self.current_fov = 0
                    self.current_t += 1

    def _finish(self):
        """Called when acquisition is complete."""
        self.timer.stop()

        # Update metadata to mark acquisition as complete (best-effort)
        try:
            self._mark_acquisition_complete()
        except Exception as e:
            print(f"Warning: Could not update acquisition metadata: {e}")

        # Always call end_zarr_acquisition, even if metadata update failed
        self.viewer.end_zarr_acquisition()
        print("Acquisition complete. Browse the dataset in the viewer.")

    def _mark_acquisition_complete(self):
        """Update zarr.json to mark acquisition as complete."""
        if self.structure == "6d":
            # 6d: fov_paths contains region paths
            zarr_dirs = list(self.fov_paths[: self.n_regions])
        else:
            # per_fov, hcs: each path is a FOV zarr
            zarr_dirs = list(self.fov_paths)

        for zarr_dir in zarr_dirs:
            zarr_json_path = zarr_dir / "zarr.json"
            if zarr_json_path.exists():
                with open(zarr_json_path, "r") as f:
                    zarr_json = json.load(f)
                attrs = zarr_json.get("attributes", {})
                if "_squid" not in attrs:
                    attrs["_squid"] = {}
                attrs["_squid"]["acquisition_complete"] = True
                zarr_json["attributes"] = attrs
                with open(zarr_json_path, "w") as f:
                    json.dump(zarr_json, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Simulate acquisition using push-based zarr API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Per-FOV zarr stores (5D: T, C, Z, Y, X per FOV)
  python simulate_zarr_acquisition.py --structure per_fov --n-fov 4

  # Single FOV (simplest)
  python simulate_zarr_acquisition.py --structure per_fov --n-fov 1

  # 6D zarr with FOV dimension (single region)
  python simulate_zarr_acquisition.py --structure 6d --n-fov 4

  # 6D multi-region (variable FOVs per region)
  python simulate_zarr_acquisition.py --structure 6d --n-regions 3 --fovs-per-region 4 6 3

  # HCS plate format
  python simulate_zarr_acquisition.py --structure hcs --wells A1 A2 B1 B2 --fov-per-well 2
""",
    )
    ap.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Output path (default: ~/Downloads/ndv_zarr_test_<timestamp>).",
    )
    ap.add_argument(
        "--structure",
        choices=["6d", "per_fov", "hcs"],
        default="per_fov",
        help="Zarr structure type (default: per_fov). "
        "6d supports multi-region via --n-regions and --fovs-per-region.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Seconds between plane writes (default: 0.05).",
    )
    ap.add_argument(
        "--n-fov",
        type=int,
        default=1,
        help="Number of FOVs (default: 1, ignored for hcs).",
    )
    ap.add_argument(
        "--n-ch", type=int, default=3, help="Number of channels (default: 3)."
    )
    ap.add_argument(
        "--n-t", type=int, default=5, help="Number of timepoints (default: 5)."
    )
    ap.add_argument(
        "--n-z", type=int, default=5, help="Number of z-levels (default: 5)."
    )
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument(
        "--channels",
        nargs="*",
        default=["DAPI", "GFP", "RFP"],
        help="Channel name strings (default: DAPI GFP RFP).",
    )
    ap.add_argument(
        "--colors",
        nargs="*",
        default=["0000FF", "00FF00", "FF0000"],
        help="Channel colors as hex RGB (default: 0000FF 00FF00 FF0000).",
    )
    ap.add_argument(
        "--pixel-size",
        type=float,
        default=0.325,
        help="Pixel size in micrometers (default: 0.325).",
    )
    ap.add_argument(
        "--z-step",
        type=float,
        default=1.5,
        help="Z step in micrometers (default: 1.5).",
    )
    # HCS-specific options
    ap.add_argument(
        "--wells",
        nargs="*",
        default=["A1", "A2", "B1", "B2"],
        help="Well IDs for HCS structure (default: A1 A2 B1 B2).",
    )
    ap.add_argument(
        "--fov-per-well",
        type=int,
        default=2,
        help="FOVs per well for HCS structure (default: 2).",
    )
    # 6d_regions-specific options
    ap.add_argument(
        "--n-regions",
        type=int,
        default=3,
        help="Number of regions for 6d_regions structure (default: 3).",
    )
    ap.add_argument(
        "--fovs-per-region",
        nargs="*",
        type=int,
        default=None,
        help="FOV count per region for 6d_regions (e.g., --fovs-per-region 4 6 3). "
        "If not specified, uses --n-fov for all regions.",
    )
    args = ap.parse_args()

    if len(args.channels) != args.n_ch:
        print(
            f"Error: --channels length ({len(args.channels)}) must match --n-ch ({args.n_ch}).",
            file=sys.stderr,
        )
        return 2

    if len(args.colors) != args.n_ch:
        print(
            f"Error: --colors length ({len(args.colors)}) must match --n-ch ({args.n_ch}).",
            file=sys.stderr,
        )
        return 2

    # Handle 6d arguments (supports multi-region)
    fovs_per_region = args.fovs_per_region
    if args.structure == "6d":
        if fovs_per_region is None:
            # Default: use n_fov for each region
            fovs_per_region = [args.n_fov] * args.n_regions
        elif len(fovs_per_region) != args.n_regions:
            print(
                f"Error: --fovs-per-region length ({len(fovs_per_region)}) must match "
                f"--n-regions ({args.n_regions}).",
                file=sys.stderr,
            )
            return 2

    if args.output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path.home() / "Downloads" / f"ndv_zarr_test_{args.structure}_{ts}"
        ).resolve()
    else:
        output_path = Path(args.output_path).expanduser().resolve()

    # Create Qt application and viewer
    app = QApplication(sys.argv)
    viewer = LightweightViewer()
    viewer.setWindowTitle(f"NDViewer Light - Zarr Simulation ({args.structure})")
    viewer.resize(1200, 800)
    viewer.show()

    # Create and start simulator
    simulator = ZarrAcquisitionSimulator(
        viewer=viewer,
        output_path=output_path,
        structure=args.structure,
        n_fov=args.n_fov,
        n_z=args.n_z,
        n_t=args.n_t,
        channels=args.channels,
        channel_colors=args.colors,
        height=args.height,
        width=args.width,
        interval_ms=int(args.interval * 1000),
        pixel_size_um=args.pixel_size,
        z_step_um=args.z_step,
        wells=args.wells,
        fov_per_well=args.fov_per_well,
        n_regions=args.n_regions,
        fovs_per_region=fovs_per_region,
    )

    # Start acquisition after event loop starts
    QTimer.singleShot(100, simulator.start)

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
