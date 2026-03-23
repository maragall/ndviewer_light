#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ERRORS=0

echo "============================================================"
echo " ndviewer_light — Full Test Suite (Linux)"
echo "============================================================"
echo ""

cd "$REPO_ROOT"

echo "[1/7] Unit tests + freeze import tests"
echo "------------------------------------------------------------"
if python -m pytest tests/ -v --tb=short 2>&1; then
    echo "OK: All unit tests passed"
else
    echo "WARN: Some unit tests failed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "[2/7] Smoke tests (import + functional checks, headless)"
echo "------------------------------------------------------------"
if xvfb-run --auto-servernum python installer/smoke_test.py 2>&1; then
    echo "OK: All smoke tests passed"
else
    echo "WARN: Some smoke tests failed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "[3/7] Import walker (static analysis)"
echo "------------------------------------------------------------"
if python installer/import_walker.py --entry installer/entry.py --spec installer/ndviewer_light_linux.spec --path . 2>&1; then
    echo "OK: Import walker passed"
else
    echo "WARN: Import walker found issues"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "[4/7] Pre-build check (Analysis phase + .so scan)"
echo "------------------------------------------------------------"
if python installer/pre_build_check.py --spec installer/ndviewer_light_linux.spec 2>&1; then
    echo "OK: Pre-build check passed"
else
    echo "WARN: Pre-build check found issues"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "[5/7] PyInstaller build (--onedir)"
echo "------------------------------------------------------------"
if bash installer/build.sh 2>&1; then
    echo "OK: PyInstaller build succeeded"
else
    echo "FAIL: PyInstaller build failed"
    ERRORS=$((ERRORS + 1))
    echo ""
    echo "============================================================"
    echo " $ERRORS test stage(s) had failures"
    echo "============================================================"
    exit $ERRORS
fi
echo ""

echo "[6/7] Scan bundled .so dependencies"
echo "------------------------------------------------------------"
if python installer/scan_libs.py dist/ndviewer_light/ 2>&1; then
    echo "OK: All .so dependencies resolved"
else
    echo "WARN: Missing .so dependencies found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "[7/7] Frozen executable smoke test"
echo "------------------------------------------------------------"
if xvfb-run --auto-servernum dist/ndviewer_light/ndviewer_light --smoke-test 2>&1; then
    echo "OK: Frozen smoke tests passed"
else
    echo "WARN: Frozen smoke tests had failures"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "============================================================"
if [ $ERRORS -eq 0 ]; then
    echo " ALL TESTS PASSED"
else
    echo " $ERRORS test stage(s) had failures"
fi
echo "============================================================"
exit $ERRORS
