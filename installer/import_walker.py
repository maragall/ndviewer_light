#!/usr/bin/env python3
"""
Static import walker for PyInstaller projects.

Traces all imports from an entry point, compares against a .spec file's
hiddenimports/excludes, and reports gaps — so you can fix them all before
spending minutes on a build.

Usage:
    python import_walker.py --entry entry.py --spec ndviewer_light.spec --path ..
    python import_walker.py --entry entry.py --spec stitcher.spec --path ../src
"""

import argparse
import ast
import importlib.util
import modulefinder
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Static import walk via modulefinder
# ---------------------------------------------------------------------------

def walk_imports(entry: Path, extra_paths: list[str]):
    """Return (found_modules, missing_modules) by tracing entry point."""
    search_path = [str(p) for p in extra_paths] + sys.path
    finder = modulefinder.ModuleFinder(path=search_path)
    try:
        finder.run_script(str(entry))
    except Exception as exc:
        print(f"  WARNING: modulefinder raised {type(exc).__name__}: {exc}")
    found = set(finder.modules.keys())
    missing = set(finder.badmodules.keys())
    return found, missing


# ---------------------------------------------------------------------------
# 2. AST-based conditional import detection
# ---------------------------------------------------------------------------

def _imports_in_node(node):
    """Yield module names from Import / ImportFrom nodes."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            yield node.module


def find_conditional_imports(source_files: list[Path]):
    """Return dict {module_name: [source_file, ...]} for imports inside try blocks."""
    conditional: dict[str, list[str]] = {}
    for src in source_files:
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        for mod in _imports_in_node(child):
                            conditional.setdefault(mod, []).append(
                                f"{src.name}:{child.lineno}"
                            )
    return conditional


# ---------------------------------------------------------------------------
# 3. Parse .spec file for hiddenimports / excludes
# ---------------------------------------------------------------------------

def parse_spec(spec_path: Path):
    """Extract hiddenimports and excludes lists from a .spec file."""
    text = spec_path.read_text(encoding="utf-8")

    def extract_list(name):
        # Match name=[...] allowing multiline
        pattern = rf"{name}\s*=\s*\[(.*?)\]"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        items = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        return items

    return extract_list("hiddenimports"), extract_list("excludes")


# ---------------------------------------------------------------------------
# 4. DLL risk check (Windows)
# ---------------------------------------------------------------------------

KNOWN_DLL_RISKS = {
    "sklearn": "msvcp140.dll / vcruntime140.dll (Win) or libgomp.so (Linux)",
    "scipy": "may bundle MKL or OpenBLAS .so/.dll files",
    "numba": "LLVM shared libraries",
    "torch": "large CUDA/cuDNN shared libraries",
    "numpy": "may bundle OpenBLAS .so on Linux",
}


def check_dll_risks(module_names: set[str]):
    """Flag modules known to carry problematic DLLs."""
    warnings = {}
    for mod in module_names:
        top = mod.split(".")[0]
        if top in KNOWN_DLL_RISKS:
            warnings[top] = KNOWN_DLL_RISKS[top]
    return warnings


# ---------------------------------------------------------------------------
# 5. Collect source files for AST scanning
# ---------------------------------------------------------------------------

def collect_source_files(extra_paths: list[Path]):
    """Collect .py files from the extra paths (project source dirs)."""
    sources = []
    for p in extra_paths:
        p = Path(p)
        if p.is_dir():
            sources.extend(p.rglob("*.py"))
        elif p.is_file() and p.suffix == ".py":
            sources.append(p)
    return sources


# ---------------------------------------------------------------------------
# 6. Transitive dependency expansion for conditional imports
# ---------------------------------------------------------------------------

def trace_transitive(module_name: str, extra_paths: list[str], depth: int = 3):
    """Try to find what a conditional import would pull in transitively."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return set()
    search_path = [str(p) for p in extra_paths] + sys.path
    finder = modulefinder.ModuleFinder(path=search_path)
    try:
        finder.run_script(spec.origin)
    except Exception:
        pass
    return set(finder.modules.keys()) | set(finder.badmodules.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Static import walker for PyInstaller")
    parser.add_argument("--entry", required=True, help="Entry point script")
    parser.add_argument("--spec", required=False, help=".spec file to compare against")
    parser.add_argument("--path", nargs="*", default=[], help="Extra search paths (like pathex)")
    args = parser.parse_args()

    entry = Path(args.entry).resolve()
    extra_paths = [Path(p).resolve() for p in args.path]

    print(f"=== Import Walker ===")
    print(f"Entry:  {entry}")
    print(f"Paths:  {[str(p) for p in extra_paths]}")
    print()

    # --- Walk imports ---
    print("Tracing imports...")
    found, missing = walk_imports(entry, extra_paths)
    print(f"  Found:   {len(found)} modules")
    print(f"  Missing: {len(missing)} modules")
    print()

    # --- Conditional imports (AST) ---
    sources = collect_source_files(extra_paths)
    sources.append(entry)
    conditional = find_conditional_imports(sources)

    # --- Parse spec ---
    hidden = []
    excludes = []
    if args.spec:
        spec_path = Path(args.spec).resolve()
        hidden, excludes = parse_spec(spec_path)
        print(f"Spec: {spec_path}")
        print(f"  hiddenimports: {len(hidden)} entries")
        print(f"  excludes:      {len(excludes)} entries")
        print()

    hidden_set = set(hidden)
    excludes_set = set(excludes)

    # --- Analysis ---

    # 1. Missing modules not in hiddenimports
    # Filter out: stdlib attrs (re.compile), platform modules (fcntl, posix),
    # internal/build modules, and things PyInstaller handles automatically.
    IGNORE_MISSING = {
        # Platform-specific (Unix/VMS/Java)
        "fcntl", "grp", "pwd", "posix", "resource", "termios", "readline",
        "java", "java.lang", "vms_lib",
        # Stdlib subattrs that modulefinder misreports
        "os.path", "collections.OrderedDict", "collections.defaultdict",
        "collections.deque", "collections.namedtuple", "collections.Counter",
        "collections.ChainMap", "collections.abc",
        "re.compile", "re.escape", "re.sub", "re.IGNORECASE",
        "asyncio.iscoroutinefunction", "http.HTTPStatus",
        "email.message_from_file",
        "ctypes.Array", "ctypes.CDLL", "ctypes.Structure", "ctypes.Union",
        "ctypes.c_char_p", "ctypes.c_ulong", "ctypes.c_void_p",
        "ctypes.cdll", "ctypes.create_string_buffer", "ctypes.sizeof",
        # Build/test infrastructure
        "distutils.filelist", "test.support._force_run",
        "packaging.licenses.canonicalize_license_expression",
    }
    # Stdlib top-level modules that PyInstaller bundles automatically
    STDLIB_AUTO = {
        "sys", "os", "re", "json", "gc", "shutil", "time", "subprocess",
        "tempfile", "traceback", "warnings", "functools", "collections",
        "ctypes", "asyncio", "http", "email", "xml", "importlib",
        "unittest", "pathlib", "threading", "logging", "typing",
        "io", "math", "struct", "hashlib", "base64", "copy",
    }
    actionable_missing = set()
    for m in missing:
        top = m.split(".")[0]
        if top.startswith("_") or top in ("encodings", "zipimport"):
            continue
        if m in IGNORE_MISSING or top in STDLIB_AUTO:
            continue
        if m in hidden_set or top in excludes_set:
            continue
        actionable_missing.add(m)

    # 2. Found modules not in hiddenimports (potential gaps — only flag local/unusual ones)
    found_not_hidden = set()
    for m in found:
        top = m.split(".")[0]
        if m in hidden_set or top in hidden_set:
            continue
        # Only flag non-stdlib modules
        try:
            spec = importlib.util.find_spec(top)
        except (ValueError, ModuleNotFoundError):
            continue
        if spec and spec.origin and ("site-packages" in str(spec.origin) or
                                      any(str(p) in str(spec.origin) for p in extra_paths)):
            found_not_hidden.add(m)

    # 3. DLL risks
    all_modules = found | missing
    dll_warnings = check_dll_risks(all_modules)

    # 4. Conditional imports analysis (skip stdlib — PyInstaller handles those)
    conditional_report = {}
    for mod, locations in conditional.items():
        top = mod.split(".")[0]
        if top in STDLIB_AUTO:
            continue  # stdlib, always available
        # Deduplicate locations
        unique_locs = sorted(set(locations))
        if top in excludes_set:
            status = "EXCLUDED (ok)"
        elif top in hidden_set or mod in hidden_set:
            status = "in hiddenimports (will be bundled)"
        else:
            status = "NOT HANDLED — add to hiddenimports or excludes"
        conditional_report[mod] = {"locations": unique_locs, "status": status}

    # --- Report ---
    print("=" * 60)
    print("REPORT")
    print("=" * 60)

    if actionable_missing:
        print("\nMISSING — imported but not found (add to hiddenimports or excludes):")
        for m in sorted(actionable_missing):
            locs = conditional.get(m, [])
            suffix = f"  (conditional: {', '.join(locs)})" if locs else ""
            print(f"  - {m}{suffix}")

    if found_not_hidden and args.spec:
        # Only show top-level unique packages
        top_pkgs = sorted({m.split(".")[0] for m in found_not_hidden} - hidden_set)
        if top_pkgs:
            print(f"\nFOUND but not in hiddenimports ({len(top_pkgs)} top-level packages):")
            for p in top_pkgs[:30]:
                print(f"  - {p}")
            if len(top_pkgs) > 30:
                print(f"  ... and {len(top_pkgs) - 30} more")

    if conditional_report:
        print("\nCONDITIONAL IMPORTS (inside try/except):")
        for mod, info in sorted(conditional_report.items()):
            print(f"  - {mod}  [{info['status']}]")
            for loc in info["locations"]:
                print(f"      at {loc}")

    if dll_warnings:
        print("\nDLL WARNINGS:")
        for pkg, warning in sorted(dll_warnings.items()):
            in_excludes = "(EXCLUDED)" if pkg in excludes_set else "(NOT excluded)"
            print(f"  - {pkg}: {warning} {in_excludes}")

    if not actionable_missing and not dll_warnings:
        print("\nAll clear — no obvious gaps detected.")

    print()
    return 1 if actionable_missing else 0


if __name__ == "__main__":
    sys.exit(main())
