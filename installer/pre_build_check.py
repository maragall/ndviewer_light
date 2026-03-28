#!/usr/bin/env python3
"""
Pre-build validator for PyInstaller projects.

Runs PyInstaller's Analysis phase (hook resolution + dependency collection)
and scans all collected .pyd/.dll files for missing DLL dependencies using
pefile. Catches hook gaps and DLL issues WITHOUT a full build.

Usage:
    cd installer
    python pre_build_check.py --spec ndviewer_light.spec
    python pre_build_check.py --spec stitcher.spec
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Run PyInstaller Analysis phase only
# ---------------------------------------------------------------------------

def run_analysis(spec_path: Path):
    """
    Run PyInstaller on a .spec file, then read back the warn file and
    collected binary list. Uses PyInstaller's own CLI to avoid config gaps.
    Runs in a temp build dir to avoid polluting the real one.
    """
    spec_path = spec_path.resolve()
    spec_dir = spec_path.parent

    with tempfile.TemporaryDirectory(prefix="pyi_check_") as tmpdir:
        workpath = os.path.join(tmpdir, "build")
        distpath = os.path.join(tmpdir, "dist")

        # Run pyinstaller — it will fail at EXE/COLLECT since we don't care
        # about those, but Analysis will complete and write the warn file + TOC
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "PyInstaller",
                str(spec_path),
                "--noconfirm",
                "--workpath", workpath,
                "--distpath", distpath,
            ],
            capture_output=True,
            text=True,
            cwd=str(spec_dir),
            timeout=300,
        )

        # Collect output for hook/analysis info
        full_output = result.stdout + "\n" + result.stderr

        # Read warn file
        warnings_text = ""
        for wf in glob.glob(os.path.join(workpath, "**/warn-*.txt"), recursive=True):
            warnings_text += Path(wf).read_text(encoding="utf-8", errors="replace")

        # Collect binaries from the dist output folder (may be in _internal/)
        binaries = []
        for ext in ("**/*.pyd", "**/*.dll"):
            for f in glob.glob(os.path.join(distpath, ext), recursive=True):
                name = os.path.basename(f)
                binaries.append((name, f, "BINARY"))

        # If temp build produced no binaries, fall back to existing dist/
        if not binaries:
            existing_dist = os.path.join(str(spec_dir), "..", "dist")
            for ext in ("**/*.pyd", "**/*.dll"):
                for f in glob.glob(os.path.join(existing_dist, ext), recursive=True):
                    name = os.path.basename(f)
                    binaries.append((name, f, "BINARY"))

        # Parse the log output for summary stats
        stats = {
            "pure": 0,
            "binaries": len(binaries),
            "datas": 0,
            "completed": "Looking for ctypes DLLs" in full_output or result.returncode == 0,
        }

        # Count from log lines
        for line in full_output.splitlines():
            if "Performing binary vs. data reclassification" in line:
                # e.g. "Performing binary vs. data reclassification (6316 entries)"
                import re as _re
                m = _re.search(r"\((\d+) entries\)", line)
                if m:
                    stats["total_entries"] = int(m.group(1))

        return binaries, warnings_text, full_output, stats


# ---------------------------------------------------------------------------
# 2. Analyze collected binaries and check DLL dependencies
# ---------------------------------------------------------------------------

def check_dll_deps(binaries):
    """
    Use pefile to scan all collected .pyd and .dll files for their
    DLL dependencies. Returns issues found.

    binaries: list of (name, src_path, typecode) tuples
    """
    try:
        import pefile
    except ImportError:
        return None, "pefile not installed — skip DLL dependency check (pip install pefile)"

    # Collect all binary paths
    collected_dlls = set()
    binary_paths = []
    for name, src_path, typecode in binaries:
        collected_dlls.add(name.lower())
        collected_dlls.add(os.path.basename(name).lower())
        if src_path and os.path.exists(src_path):
            binary_paths.append((name, src_path))

    # System DLLs that Windows provides (don't need to be bundled)
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
        # MSVC runtime — usually redistributed or on system
        "vcruntime140.dll", "vcruntime140_1.dll",
        "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
        "concrt140.dll", "vcomp140.dll",
        "ucrtbased.dll", "vcruntime140d.dll",
        # API sets (Windows 10+)
        "api-ms-win-",
    }

    def is_system_dll(name):
        name_lower = name.lower()
        if name_lower in SYSTEM_DLLS:
            return True
        if name_lower.startswith("api-ms-win-"):
            return True
        return False

    issues = []
    checked = 0
    for name, src_path in binary_paths:
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in (".pyd", ".dll"):
            continue
        checked += 1
        try:
            pe = pefile.PE(src_path, fast_load=True)
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
            if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                pe.close()
                continue
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dep_dll = entry.dll.decode("utf-8", errors="replace")
                dep_lower = dep_dll.lower()
                if is_system_dll(dep_lower):
                    continue
                # Check if this DLL is in our collected set
                if dep_lower not in collected_dlls:
                    # Check with python3X.dll pattern
                    if dep_lower.startswith("python3") and dep_lower.endswith(".dll"):
                        continue  # Python DLL, handled by bootloader
                    issues.append({
                        "binary": name,
                        "source": src_path,
                        "missing_dll": dep_dll,
                    })
            pe.close()
        except Exception as exc:
            issues.append({
                "binary": name,
                "source": src_path,
                "error": str(exc),
            })

    return issues, f"Scanned {checked} .pyd/.dll files"


# ---------------------------------------------------------------------------
# 3. Parse warnings for actionable items
# ---------------------------------------------------------------------------

def parse_warnings(warnings_text: str):
    """Extract actionable warnings from PyInstaller warn file."""
    if not warnings_text:
        return [], []

    critical = []  # top-level missing imports (will crash)
    notable = []   # delayed/optional (may be fine)

    for line in warnings_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("missing module named"):
            continue

        # Format: "missing module named X - imported by Y (context)"
        # context can be: top-level, conditional, delayed, optional, delayed/conditional
        if "(top-level)" in line:
            critical.append(line)
        elif "(delayed" in line or "(optional" in line or "(conditional" in line:
            notable.append(line)
        else:
            notable.append(line)

    return critical, notable


# ---------------------------------------------------------------------------
# 4. Check which packages have PyInstaller hooks
# ---------------------------------------------------------------------------

def check_hooks(hiddenimports: list[str]):
    """Check which of our dependencies have PyInstaller hooks."""
    try:
        import PyInstaller
        hooks_dir = Path(PyInstaller.__path__[0]) / "hooks"
    except (ImportError, IndexError):
        return {}

    available_hooks = set()
    for f in hooks_dir.glob("hook-*.py"):
        # hook-scipy.py -> scipy
        hook_name = f.stem.replace("hook-", "")
        available_hooks.add(hook_name)

    results = {}
    seen_top = set()
    for mod in hiddenimports:
        top = mod.split(".")[0]
        if top in seen_top:
            continue
        seen_top.add(top)
        # Check for exact match or top-level match
        has_hook = (top in available_hooks or
                    mod in available_hooks or
                    any(h.startswith(top + ".") or h == top for h in available_hooks))
        results[top] = has_hook

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-build validator: runs PyInstaller Analysis + DLL scan"
    )
    parser.add_argument("--spec", required=True, help="Path to .spec file")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    print(f"{'=' * 60}")
    print(f"PRE-BUILD CHECK: {spec_path.name}")
    print(f"{'=' * 60}")
    print()

    # --- Parse spec for hiddenimports/excludes ---
    from import_walker import parse_spec
    hidden, excludes = parse_spec(spec_path)
    print(f"Spec: {len(hidden)} hiddenimports, {len(excludes)} excludes")

    # --- Check hook coverage ---
    print("\n--- HOOK COVERAGE ---")
    hook_status = check_hooks(hidden)
    no_hook = [pkg for pkg, has in hook_status.items() if not has]
    has_hook = [pkg for pkg, has in hook_status.items() if has]
    print(f"  With hooks ({len(has_hook)}): {', '.join(sorted(has_hook))}")
    if no_hook:
        print(f"  WITHOUT hooks ({len(no_hook)}): {', '.join(sorted(no_hook))}")
        print(f"  (These rely entirely on hiddenimports + static tracing)")

    # --- Run Analysis ---
    print("\n--- PYINSTALLER ANALYSIS ---")
    print("Running full PyInstaller build in temp dir (this may take 1-3 min)...")
    try:
        binaries, warnings_text, full_output, stats = run_analysis(spec_path)
    except Exception as exc:
        print(f"\n  ANALYSIS FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    if not stats["completed"]:
        print("  WARNING: Analysis may not have completed fully")

    print(f"  Binaries collected for DLL scan: {len(binaries)}")
    if "total_entries" in stats:
        print(f"  Total entries reclassified:      {stats['total_entries']}")

    # --- Parse warnings ---
    critical_warns, notable_warns = parse_warnings(warnings_text)
    if critical_warns:
        print(f"\n  CRITICAL WARNINGS ({len(critical_warns)} top-level missing imports):")
        for w in critical_warns[:20]:
            print(f"    {w}")
        if len(critical_warns) > 20:
            print(f"    ... and {len(critical_warns) - 20} more")
    else:
        print(f"\n  No critical (top-level) missing imports.")

    if notable_warns:
        # Filter to only show non-stdlib, non-excluded
        filtered = []
        for w in notable_warns:
            skip = False
            for ex in excludes:
                if ex in w:
                    skip = True
                    break
            if not skip:
                filtered.append(w)
        if filtered:
            print(f"\n  Notable warnings ({len(filtered)} delayed/optional):")
            for w in filtered[:15]:
                print(f"    {w}")
            if len(filtered) > 15:
                print(f"    ... and {len(filtered) - 15} more")

    # --- DLL dependency scan ---
    print(f"\n--- DLL DEPENDENCY SCAN ---")
    dll_issues, dll_summary = check_dll_deps(binaries)
    print(f"  {dll_summary}")

    if dll_issues is None:
        print(f"  {dll_summary}")  # error message
    elif dll_issues:
        # Group by missing DLL
        by_dll = {}
        errors = []
        for issue in dll_issues:
            if "error" in issue:
                errors.append(issue)
            else:
                dll = issue["missing_dll"]
                by_dll.setdefault(dll, []).append(issue["binary"])

        if by_dll:
            print(f"\n  MISSING DLLs ({len(by_dll)} unique):")
            for dll, binaries in sorted(by_dll.items()):
                print(f"    {dll}")
                for b in binaries[:3]:
                    print(f"      needed by: {b}")
                if len(binaries) > 3:
                    print(f"      ... and {len(binaries) - 3} more")

        if errors:
            print(f"\n  Scan errors ({len(errors)}):")
            for e in errors[:5]:
                print(f"    {e['binary']}: {e['error']}")
    else:
        print(f"  All DLL dependencies resolved.")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    has_issues = bool(critical_warns) or bool(dll_issues)
    if has_issues:
        print("RESULT: Issues found — review above before building")
    else:
        print("RESULT: No critical issues detected — ready to build")
    print(f"{'=' * 60}")

    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
