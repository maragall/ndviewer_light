"""
Simulate a live acquisition using the push-based API.

Unlike simulate_live_acquisition.py (file-based mode), this script:
- Runs the viewer in-process
- Calls start_acquisition() to configure the viewer
- Calls register_image() for each saved image
- Does not rely on filesystem polling

This tests the push-based API as used by the Squid acquisition system.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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


def _atomic_tiff_write(path: Path, image: np.ndarray) -> None:
    """Write a TIFF via a temp file + atomic replace to avoid partial reads."""
    import tifffile as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tf.imwrite(str(tmp), image, photometric="minisblack")
        os.replace(str(tmp), str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass  # Best effort cleanup
        raise


class AcquisitionSimulator:
    """Simulates acquisition by writing files and calling register_image()."""

    def __init__(
        self,
        viewer: LightweightViewer,
        root: Path,
        n_fov: int,
        n_z: int,
        n_t: int,
        channels: list[str],
        height: int,
        width: int,
        interval_ms: int,
    ):
        self.viewer = viewer
        self.root = root
        self.n_fov = n_fov
        self.n_z = n_z
        self.n_t = n_t
        self.channels = channels
        self.height = height
        self.width = width
        self.interval_ms = interval_ms

        # Current position in acquisition
        self.current_t = 0
        self.current_fov = 0

        # Precompute base image pattern
        y = np.arange(height, dtype=np.uint16)[:, None]
        x = np.arange(width, dtype=np.uint16)[None, :]
        self.base = y + x

        # Generate FOV labels (well:fov format)
        self.fov_labels = []
        wells = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
        fov_per_well = (n_fov + len(wells) - 1) // len(wells)
        for i in range(n_fov):
            well_idx = i // fov_per_well
            fov_in_well = i % fov_per_well
            well = wells[well_idx % len(wells)]
            self.fov_labels.append(f"{well}:{fov_in_well}")

        # Timer for periodic writes
        self.timer = QTimer()
        self.timer.timeout.connect(self._write_next_fov)

    def start(self):
        """Start the simulated acquisition."""
        print(f"Starting push-based acquisition simulation")
        print(f"  Output: {self.root}")
        print(
            f"  FOVs: {self.n_fov}, Z: {self.n_z}, T: {self.n_t}, Channels: {self.channels}"
        )

        # Configure viewer via push-based API
        self.viewer.start_acquisition(
            channels=self.channels,
            num_z=self.n_z,
            height=self.height,
            width=self.width,
            fov_labels=self.fov_labels,
        )

        # Start writing
        self.timer.start(self.interval_ms)

    def _write_next_fov(self):
        """Write all z-planes and channels for current FOV, then advance."""
        if self.current_t >= self.n_t:
            self._finish()
            return

        t = self.current_t
        fov = self.current_fov
        tp_dir = self.root / str(t)
        tp_dir.mkdir(parents=True, exist_ok=True)

        fov_label = self.fov_labels[fov]
        well, fov_in_well = fov_label.split(":")

        for z in range(self.n_z):
            for c, ch_name in enumerate(self.channels):
                # Create image with identifying pattern
                offset = np.uint16(t * 97 + fov * 11 + c * 301 + z * 50)
                img = (self.base + offset).astype(np.uint16, copy=True)

                # Overlay text label
                label = f"T={t:02d} F={fov} Z={z:02d} C={c}"
                _draw_text(img, label, x=20, y=20, scale=10, value=60000)

                # Write file
                fname = f"{well}_{fov_in_well}_{z}_{ch_name}.tiff"
                filepath = tp_dir / fname
                _atomic_tiff_write(filepath, img)

                # Register with viewer (push-based API)
                self.viewer.register_image(
                    t=t,
                    fov_idx=fov,
                    z=z,
                    channel=ch_name,
                    filepath=str(filepath),
                )

        print(
            f"[t={t}] Wrote FOV {fov} ({fov_label}): {self.n_z} z × {len(self.channels)} ch"
        )

        # Advance to next FOV
        self.current_fov += 1
        if self.current_fov >= self.n_fov:
            self.current_fov = 0
            self.current_t += 1

    def _finish(self):
        """Called when acquisition is complete."""
        self.timer.stop()
        self.viewer.end_acquisition()
        print("Acquisition complete. Browse the dataset in the viewer.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Simulate acquisition using push-based API"
    )
    ap.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Output dataset folder (default: ~/Downloads/ndv_push_test_<timestamp>).",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between FOV writes (default: 0.1).",
    )
    ap.add_argument("--n-fov", type=int, default=20)
    ap.add_argument("--n-ch", type=int, default=3)
    ap.add_argument("--n-t", type=int, default=5)
    ap.add_argument("--n-z", type=int, default=5)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument(
        "--channels",
        nargs="*",
        default=["BF", "DAPI", "GFP"],
        help="Channel name strings.",
    )
    args = ap.parse_args()

    if len(args.channels) != args.n_ch:
        print(
            f"Error: --channels length ({len(args.channels)}) must match --n-ch ({args.n_ch}).",
            file=sys.stderr,
        )
        return 2

    if args.dataset_root is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        root = (Path.home() / "Downloads" / f"ndv_push_test_{ts}").resolve()
    else:
        root = Path(args.dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Create Qt application and viewer
    app = QApplication(sys.argv)
    viewer = LightweightViewer()
    viewer.setWindowTitle("NDViewer Light - Push-Based Simulation")
    viewer.resize(1200, 800)
    viewer.show()

    # Create and start simulator
    simulator = AcquisitionSimulator(
        viewer=viewer,
        root=root,
        n_fov=args.n_fov,
        n_z=args.n_z,
        n_t=args.n_t,
        channels=args.channels,
        height=args.height,
        width=args.width,
        interval_ms=int(args.interval * 1000),
    )

    # Start acquisition after event loop starts
    QTimer.singleShot(100, simulator.start)

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
