# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for ndviewer_light (Linux / AppImage).

IMPORTANT: Run from the installer/ directory:
  cd installer && python -m PyInstaller ndviewer_light_linux.spec --noconfirm
"""

import os
import subprocess
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

vispy_datas = collect_data_files('vispy')

# Collect ALL submodules so PyInstaller bundles entire packages
ndviewer_light_imports = collect_submodules('ndviewer_light')
ndv_imports = collect_submodules('ndv')
vispy_imports = collect_submodules('vispy')
cmap_imports = collect_submodules('cmap')
psygnal_imports = collect_submodules('psygnal')

# ---------------------------------------------------------------------------
# Bundle xcb platform plugin dependencies from the system.
# These .so files are required by Qt's libqxcb.so but are not inside
# the PyQt5 wheel, so PyInstaller doesn't collect them automatically.
# ---------------------------------------------------------------------------
xcb_libs = []
for lib_name in [
    'libxcb-icccm', 'libxcb-image', 'libxcb-keysyms',
    'libxcb-randr', 'libxcb-render-util', 'libxcb-xinerama',
    'libxcb-xfixes', 'libxcb-shape', 'libxkbcommon-x11', 'libxkbcommon',
]:
    result = subprocess.run(
        ['find', '/usr/lib', '-name', f'{lib_name}*.so*', '-type', 'f'],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().split('\n'):
        if line and os.path.isfile(line):
            xcb_libs.append((line, '.'))

a = Analysis(
    ['entry.py'],
    pathex=[os.path.abspath('..')],
    binaries=xcb_libs,
    datas=vispy_datas,
    hiddenimports=ndviewer_light_imports + ndv_imports + vispy_imports + cmap_imports + psygnal_imports + [
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'superqt',
        'superqt.sliders',
        'superqt.iconify',
        'pyconify',
        'dask',
        'dask.array',
        'tifffile',
        'xarray',
        'scipy',
        'scipy.ndimage',
        'vispy',
        'vispy.app',
        'vispy.app.backends._pyqt5',
        'vispy.visuals.volume',
        'tensorstore',
        'ml_dtypes',
        'zarr',
        'zarr.storage',
        'OpenGL',
        'OpenGL.GL',
        'OpenGL.platform.glx',
        'OpenGL.platform.egl',
        'numpy.core._methods',
        'numpy.lib.format',
        'xml.etree.ElementTree',
        'importlib.metadata',
        # Optional dependencies (bundle when available)
        'PIL',
        'colorspacious',
        'viscm',
        'numba',
        'numba.core',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ndviewer_light',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    # icon='ndviewer_light.png',  # TODO: add proper icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,  # DO NOT strip — corrupts scipy openblas .so files (ELF page alignment)
    upx=True,
    upx_exclude=[
        'libscipy_openblas*.so*',
        'libopenblas*.so*',
    ],
    name='ndviewer_light',
)
