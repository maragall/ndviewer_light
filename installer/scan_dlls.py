#!/usr/bin/env python3
"""Scan a PyInstaller dist folder for missing DLL dependencies using pefile."""

import argparse
import glob
import os
import sys


SYSTEM_DLLS = {
    "kernel32.dll", "user32.dll", "gdi32.dll", "advapi32.dll",
    "shell32.dll", "ole32.dll", "oleaut32.dll", "comctl32.dll",
    "comdlg32.dll", "ws2_32.dll", "wsock32.dll", "ntdll.dll",
    "msvcrt.dll", "ucrtbase.dll", "bcrypt.dll", "crypt32.dll",
    "secur32.dll", "winspool.drv", "shlwapi.dll", "rpcrt4.dll",
    "imm32.dll", "winmm.dll", "version.dll", "netapi32.dll",
    "userenv.dll", "setupapi.dll", "cfgmgr32.dll", "powrprof.dll",
    "mswsock.dll", "iphlpapi.dll", "wldap32.dll", "normaliz.dll",
    "dnsapi.dll", "dbghelp.dll", "psapi.dll", "pdh.dll",
    "vcruntime140.dll", "vcruntime140_1.dll",
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "concrt140.dll", "vcomp140.dll",
    "ucrtbased.dll", "vcruntime140d.dll",
    "opengl32.dll", "glu32.dll", "d3d11.dll", "dxgi.dll",
    "dwmapi.dll", "uxtheme.dll", "propsys.dll", "shcore.dll",
    "wtsapi32.dll", "ncrypt.dll",
    "bcryptprimitives.dll", "d3d9.dll", "imagehlp.dll", "mpr.dll",
    "d3d11.dll", "dxgi.dll", "d3d12.dll", "dxcore.dll",
    "oleacc.dll", "uiautomationcore.dll", "credui.dll",
    "cryptui.dll", "wevtapi.dll", "cabinet.dll",
    "d2d1.dll", "dwrite.dll", "winhttp.dll", "wininet.dll",
    "webservices.dll",
}


def main():
    parser = argparse.ArgumentParser(description="Scan dist folder for missing DLLs")
    parser.add_argument("dist_dir", help="Path to dist/<appname>/_internal or dist/<appname>")
    args = parser.parse_args()

    try:
        import pefile
    except ImportError:
        print("ERROR: pefile not installed. Run: pip install pefile")
        return 1

    dist_dir = args.dist_dir

    # Collect all .pyd and .dll files
    files = (
        glob.glob(os.path.join(dist_dir, "**/*.pyd"), recursive=True)
        + glob.glob(os.path.join(dist_dir, "**/*.dll"), recursive=True)
    )

    if not files:
        print(f"No .pyd/.dll files found in {dist_dir}")
        return 1

    # Build set of all DLLs present in the bundle
    collected = set()
    for f in files:
        collected.add(os.path.basename(f).lower())

    print(f"Scanning {len(files)} files ({len(collected)} unique) in {dist_dir}")
    print()

    issues = {}
    errors = []
    scanned = 0

    for f in files:
        scanned += 1
        try:
            pe = pefile.PE(f, fast_load=True)
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
            if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                pe.close()
                continue
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dep = entry.dll.decode("utf-8", errors="replace").lower()
                if dep in SYSTEM_DLLS:
                    continue
                if dep.startswith("api-ms-win-"):
                    continue
                if dep.startswith("python3"):
                    continue
                if dep not in collected:
                    issues.setdefault(dep, []).append(os.path.basename(f))
            pe.close()
        except Exception as e:
            errors.append(f"  {os.path.basename(f)}: {e}")

    print(f"Scanned: {scanned} files")

    if issues:
        print(f"\nMISSING DLLs ({len(issues)} unique):")
        for dll, needed_by in sorted(issues.items()):
            print(f"  {dll}")
            for b in sorted(set(needed_by))[:5]:
                print(f"    needed by: {b}")
            if len(needed_by) > 5:
                print(f"    ... and {len(needed_by) - 5} more")
    else:
        print("\nAll DLL dependencies resolved!")

    if errors:
        print(f"\nScan errors ({len(errors)}):")
        for e in errors[:10]:
            print(e)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
