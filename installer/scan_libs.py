#!/usr/bin/env python3
"""Scan a PyInstaller dist folder for missing shared library dependencies using ldd.

Linux equivalent of scan_dlls.py (which uses pefile for Windows .pyd/.dll files).
"""

import argparse
import glob
import os
import subprocess
import sys


# System-provided libraries that should NOT be bundled (they come from the OS / GPU driver)
SYSTEM_LIBS = {
    "linux-vdso.so.1",
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libm.so.6",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
    "libutil.so.1",
    "libresolv.so.2",
    "libnsl.so.1",
    "libcrypt.so.1",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    # GPU driver libs — must use system's, not bundled
    "libGL.so.1",
    "libGLX.so.0",
    "libGLdispatch.so.0",
    "libEGL.so.1",
    "libOpenGL.so.0",
    "libGLESv2.so.2",
    # X11 core (always present on desktop Linux)
    "libX11.so.6",
    "libXext.so.6",
    "libXrender.so.1",
    "libXi.so.6",
    "libXfixes.so.3",
    "libXcursor.so.1",
    "libXrandr.so.2",
    "libXcomposite.so.1",
    "libXdamage.so.1",
    "libxcb.so.1",
}


def is_system_lib(name):
    """Check if a library name is a system-provided lib."""
    basename = os.path.basename(name)
    if basename in SYSTEM_LIBS:
        return True
    # Check without version suffix (e.g., libc.so.6 matches libc.so)
    for sys_lib in SYSTEM_LIBS:
        base = sys_lib.split(".so")[0]
        if basename.startswith(base + ".so"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Scan dist folder for missing .so dependencies")
    parser.add_argument("dist_dir", help="Path to dist/<appname>/ or dist/<appname>/_internal")
    args = parser.parse_args()

    dist_dir = args.dist_dir

    # Collect all .so files
    files = glob.glob(os.path.join(dist_dir, "**/*.so*"), recursive=True)
    if not files:
        print(f"No .so files found in {dist_dir}")
        return 1

    # Build set of all .so basenames present in the bundle
    collected = set()
    for f in files:
        collected.add(os.path.basename(f))

    print(f"Scanning {len(files)} .so files ({len(collected)} unique) in {dist_dir}")
    print()

    issues = {}
    errors = []
    scanned = 0

    for f in files:
        scanned += 1
        try:
            result = subprocess.run(
                ["ldd", f],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if "not found" in line:
                    # Format: "libfoo.so.1 => not found"
                    dep = line.split("=>")[0].strip()
                    if not is_system_lib(dep):
                        issues.setdefault(dep, []).append(os.path.basename(f))
        except subprocess.TimeoutExpired:
            errors.append(f"  {os.path.basename(f)}: ldd timed out")
        except Exception as e:
            errors.append(f"  {os.path.basename(f)}: {e}")

    print(f"Scanned: {scanned} files")

    if issues:
        print(f"\nMISSING SHARED LIBRARIES ({len(issues)} unique):")
        for lib, needed_by in sorted(issues.items()):
            print(f"  {lib}")
            for b in sorted(set(needed_by))[:5]:
                print(f"    needed by: {b}")
            if len(needed_by) > 5:
                print(f"    ... and {len(needed_by) - 5} more")
    else:
        print("\nAll shared library dependencies resolved!")

    if errors:
        print(f"\nScan errors ({len(errors)}):")
        for e in errors[:10]:
            print(e)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
