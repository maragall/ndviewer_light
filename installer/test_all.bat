@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  ndviewer_light — Full Test Suite
echo ============================================================
echo.

set REPO=C:\Users\peter\Downloads\ndviewer_light
set ERRORS=0

call conda activate ndviewer_light 2>nul
if errorlevel 1 (
    echo No ndviewer_light conda env found. Creating from scratch...
    call conda env create -f %REPO%\environment.yml -y
    call conda activate ndviewer_light
)

cd /d %REPO%

echo [1/5] Unit tests (215 existing + 16 freeze imports)
echo ------------------------------------------------------------
python -m pytest tests/ -v --tb=short 2>&1
if errorlevel 1 (
    echo WARN: Some unit tests failed
    set /a ERRORS+=1
) else (
    echo OK: All unit tests passed
)
echo.

echo [2/5] Smoke tests (import + functional checks, headless)
echo ------------------------------------------------------------
python installer/smoke_test.py 2>&1
if errorlevel 1 (
    echo WARN: Some smoke tests failed
    set /a ERRORS+=1
) else (
    echo OK: All smoke tests passed
)
echo.

echo [3/5] Functional GUI tests (headless, real datasets)
echo ------------------------------------------------------------
python installer/functional_test.py 2>&1
if errorlevel 1 (
    echo WARN: Some functional tests failed
    set /a ERRORS+=1
) else (
    echo OK: All functional tests passed
)
echo.

echo [4/5] PyInstaller build
echo ------------------------------------------------------------
where pyinstaller >nul 2>&1
if errorlevel 1 (
    pip install pyinstaller -q
)
pyinstaller installer/ndviewer_light.spec --noconfirm 2>&1
if errorlevel 1 (
    echo FAIL: PyInstaller build failed
    set /a ERRORS+=1
    goto :summary
) else (
    echo OK: PyInstaller build succeeded
)
echo.

echo [5/5] Frozen exe smoke test
echo ------------------------------------------------------------
dist\ndviewer_light\ndviewer_light.exe --smoke-test 2>&1
if errorlevel 1 (
    echo WARN: Frozen smoke tests had failures
    set /a ERRORS+=1
) else (
    echo OK: Frozen smoke tests passed
)
echo.

:summary
echo ============================================================
if %ERRORS%==0 (
    echo  ALL TESTS PASSED
) else (
    echo  %ERRORS% test stage(s) had failures
)
echo ============================================================
exit /b %ERRORS%
