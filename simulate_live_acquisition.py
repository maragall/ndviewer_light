"""
Simulate an ongoing acquisition on disk and launch ndviewer_light on it.

Writes a "single_tiff" style dataset:

  <dataset_root>/
    0/
      R0_<fov>_<z>_<channel>.tiff
    1/
      ...

The viewer is expected to detect new timepoints (new numeric folders) and
optionally new FOVs as they appear in timepoint 0.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


_FONT_5X7: dict[str, list[str]] = {
    # 5x7 bitmap font, '#' = on, '.' = off
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
    "=": [".....", "#####", ".....", "#####", ".....", ".....", "....."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],
}


def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int, value: int) -> None:
    """
    Draw text into a uint16 image in-place using the bitmap font above.
    Text is clipped if it goes out of bounds.
    """
    h, w = img.shape
    cursor_x = x
    cursor_y = y
    char_w = 5 * scale
    char_h = 7 * scale
    spacing = 1 * scale

    for ch in text:
        glyph = _FONT_5X7.get(ch)
        if glyph is None:
            glyph = _FONT_5X7[" "]

        # Newline support (not currently used, but handy)
        if ch == "\n":
            cursor_x = x
            cursor_y += char_h + spacing
            continue

        # Clip quickly if fully out of bounds
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
    tf.imwrite(str(tmp), image, photometric="minisblack")
    os.replace(str(tmp), str(path))


def _make_plane(base: np.ndarray, t: int, fov: int, c: int) -> np.ndarray:
    # Deterministic, fast, moderately compressible pattern.
    # Keep values in uint16.
    img = (base + np.uint16(t * 97 + fov * 11 + c * 301)).astype(np.uint16, copy=False)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Output dataset folder (default: ~/Downloads/ndv_live_test_<timestamp>).",
    )
    ap.add_argument("--interval", type=float, default=0.5, help="Seconds between writes (default: 0.5).")
    ap.add_argument("--n-fov", type=int, default=20)
    ap.add_argument("--n-ch", type=int, default=3)
    ap.add_argument("--n-t", type=int, default=25)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument(
        "--channels",
        nargs="*",
        default=["405nm", "488nm", "561nm"],
        help="Channel name strings used in filenames.",
    )
    ap.add_argument("--region", default="R0")
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument(
        "--fovs-per-tick-in-t0",
        type=int,
        default=5,
        help="During initial phase, fill timepoint 0 with this many new FOVs per tick (default: 5).",
    )
    ap.add_argument(
        "--no-launch",
        action="store_true",
        help="Don't launch the viewer; only write data.",
    )
    args = ap.parse_args()

    if args.dataset_root is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        root = (Path.home() / "Downloads" / f"ndv_live_test_{ts}").resolve()
    else:
        root = Path(args.dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if len(args.channels) != args.n_ch:
        print(f"Error: --channels length ({len(args.channels)}) must match --n-ch ({args.n_ch}).", file=sys.stderr)
        return 2

    # Launch viewer as a separate process so we can keep writing.
    viewer_proc = None
    if not args.no_launch:
        viewer_cmd = [sys.executable, str(Path(__file__).parent / "ndviewer_light.py"), str(root)]
        print("Launching viewer:", " ".join(viewer_cmd))
        viewer_proc = subprocess.Popen(viewer_cmd)

        # Give the viewer a moment to initialize before writing lots of files.
        time.sleep(0.75)

    # Precompute a base ramp once (fast to derive planes from it).
    y = np.arange(args.height, dtype=np.uint16)[:, None]
    x = np.arange(args.width, dtype=np.uint16)[None, :]
    base = (y + x)  # uint16 wrap is fine

    print(f"Writing dataset to: {root}")
    print(f"Plan: n_fov={args.n_fov}, n_ch={args.n_ch}, n_t={args.n_t}, size={args.height}x{args.width}")
    print(f"Tick interval: {args.interval}s")

    # Phase 1: simulate "incomplete acquisition" where FOVs appear gradually in timepoint 0.
    fov_written_t0 = 0
    fovs_per_tick = max(1, int(args.fovs_per_tick_in_t0))
    while fov_written_t0 < args.n_fov:
        t = 0
        tp_dir = root / str(t)
        tp_dir.mkdir(parents=True, exist_ok=True)

        end = min(args.n_fov, fov_written_t0 + fovs_per_tick)
        for fov in range(fov_written_t0, end):
            for c, ch_name in enumerate(args.channels):
                fname = f"{args.region}_{fov}_{args.z}_{ch_name}.tiff"
                out = tp_dir / fname
                img = _make_plane(base, t=t, fov=fov, c=c).copy()
                # Overlay "T=<t> FOV=<fov> CH=<idx>" into the pixels
                label = f"T={t:02d} FOV={fov:02d} CH={c}"
                _draw_text(img, label, x=20, y=20, scale=10, value=60000)
                _atomic_tiff_write(out, img)
        fov_written_t0 = end
        print(f"[t=0] wrote FOVs: 0..{fov_written_t0-1} (of {args.n_fov})")
        time.sleep(args.interval)

    # Phase 2: write full timepoints t=1..n_t-1, complete per tick.
    for t in range(1, args.n_t):
        tp_dir = root / str(t)
        tp_dir.mkdir(parents=True, exist_ok=True)
        for fov in range(args.n_fov):
            for c, ch_name in enumerate(args.channels):
                fname = f"{args.region}_{fov}_{args.z}_{ch_name}.tiff"
                out = tp_dir / fname
                img = _make_plane(base, t=t, fov=fov, c=c).copy()
                label = f"T={t:02d} FOV={fov:02d} CH={c}"
                _draw_text(img, label, x=20, y=20, scale=10, value=60000)
                _atomic_tiff_write(out, img)
        print(f"[t={t}] wrote all planes ({args.n_fov} fov × {args.n_ch} ch)")
        time.sleep(args.interval)

    print("Done writing. Leave the viewer open to browse the dataset.")
    if viewer_proc is not None:
        print("Viewer PID:", viewer_proc.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


