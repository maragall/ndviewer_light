"""Manual test for 3D volume visualization with anisotropic voxels.

This test generates synthetic 3D data with known structures to verify
that the Z aspect ratio correction works correctly in volume rendering.

NOT run during CI - use for manual visual testing only.

Usage:
    python tests/test_3d_visualization.py

Cleanup:
    Test data is saved to a temporary directory (e.g., /tmp/ndv_3d_test_*)
    and kept after the viewer closes to allow inspection.

    To clean up old test directories:
        rm -rf /tmp/ndv_3d_test_*              # remove all
        find /tmp -maxdepth 1 -type d -name 'ndv_3d_test_*' -mtime +7 -exec rm -rf {} +  # remove >7 days old
"""

import numpy as np
import tifffile
import json
from pathlib import Path
import tempfile
import sys


def generate_3d_test_volume(
    nx: int = 100,
    ny: int = 100,
    nz: int = 20,
    pixel_xy: float = 1.0,
    pixel_z: float = 3.0,
) -> tuple:
    """Generate a 3D test volume with spheres, helix, and tube structures.

    Args:
        nx, ny, nz: Volume dimensions in pixels
        pixel_xy: XY pixel size in micrometers
        pixel_z: Z step size in micrometers

    Returns:
        Tuple of (volume as uint16 array, tmp_dir Path)
    """
    # Create output directory with proper structure for ndviewer_light
    tmp_dir = Path(tempfile.mkdtemp(prefix="ndv_3d_test_"))
    subdir = tmp_dir / "0"
    subdir.mkdir()

    # Create coordinate grids in physical units
    x = np.arange(nx) * pixel_xy
    y = np.arange(ny) * pixel_xy
    z = np.arange(nz) * pixel_z
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    volume = np.zeros((nx, ny, nz), dtype=np.float32)

    # 1. Add spheres at different depths (like cell nuclei)
    spheres = [
        (25, 25, 15, 12, 1.0),  # (cx_um, cy_um, cz_um, radius_um, intensity)
        (75, 30, 24, 10, 0.8),
        (50, 70, 36, 14, 1.2),
        (30, 60, 45, 8, 0.6),
        (70, 75, 12, 11, 0.9),
    ]
    for cx, cy, cz, r, intensity in spheres:
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        sphere_signal = intensity * np.exp(-0.5 * (dist / (r * 0.5)) ** 2)
        sphere_signal[dist > r * 2] = 0
        volume += sphere_signal

    # 2. Add a helix structure (like DNA or spiral vessel)
    t = np.linspace(0, 4 * np.pi, 1000)
    helix_x = 50 + 20 * np.cos(t)
    helix_y = 50 + 20 * np.sin(t)
    helix_z = np.linspace(6, 54, len(t))
    for hx, hy, hz in zip(helix_x, helix_y, helix_z):
        dist = np.sqrt((X - hx) ** 2 + (Y - hy) ** 2 + (Z - hz) ** 2)
        helix_signal = 0.7 * np.exp(-0.5 * (dist / 3) ** 2)
        helix_signal[dist > 9] = 0
        volume += helix_signal

    # 3. Add a vertical hollow tube (like a blood vessel)
    # Use physical coordinates for consistency with spheres and helix
    dist_to_axis = np.sqrt((X - 15) ** 2 + (Y - 85) ** 2)
    tube_signal = 0.5 * np.exp(-0.5 * ((dist_to_axis - 8) / 2) ** 2)
    volume += tube_signal

    # Transpose to (z, y, x) for saving
    volume = volume.transpose(2, 1, 0)

    # Normalize to use ~75% of dynamic range (avoid saturation in volume rendering)
    volume = (volume / volume.max()) * 50000
    volume = volume.astype(np.uint16)

    # Save as individual TIFFs
    for z_idx in range(nz):
        tiff_path = subdir / f"A1_0_{z_idx}_GFP.tiff"
        tifffile.imwrite(str(tiff_path), volume[z_idx])

    # Create acquisition parameters
    params = {"pixel_size_um": pixel_xy, "dz_um": pixel_z}
    with open(tmp_dir / "acquisition_parameters.json", "w") as f:
        json.dump(params, f)

    return volume, tmp_dir


def main():
    """Generate test data and launch viewer."""
    print("Generating 3D test volume...")
    print("  Dimensions: 100x100x20")
    print("  Pixel sizes: 1um x 1um x 3um")
    print("  Z aspect ratio: 3x")
    print()

    volume, tmp_dir = generate_3d_test_volume()

    print(f"Volume shape: {volume.shape}")
    print(f"Value range: {volume.min()} - {volume.max()}")
    print(f"Mean intensity: {volume.mean():.0f}")
    print(f"Dataset saved to: {tmp_dir}")
    print()

    # Try to launch viewer
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ndviewer_light import LightweightMainWindow
        from PyQt5.QtWidgets import QApplication

        print("Launching viewer...")
        print("  - Check 2D slices show spheres, helix, and tube")
        print("  - Switch to 3D mode to verify Z aspect ratio (should be 3x)")
        print()

        app = QApplication(sys.argv)
        window = LightweightMainWindow(str(tmp_dir))
        window.show()
        sys.exit(app.exec_())

    except ImportError as e:
        print(f"Could not launch viewer: {e}")
        print(f"Run manually: python ndviewer_light.py {tmp_dir}")


if __name__ == "__main__":
    main()
