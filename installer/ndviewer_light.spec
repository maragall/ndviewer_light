# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for ndviewer_light.

IMPORTANT: Run from the installer/ directory:
  cd installer && python -m PyInstaller ndviewer_light.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

vispy_datas = collect_data_files('vispy')

# Collect ALL submodules of ndviewer_light so the package is fully bundled
ndviewer_light_imports = collect_submodules('ndviewer_light')

a = Analysis(
    ['entry.py'],
    pathex=[os.path.abspath('..')],
    binaries=[],
    datas=vispy_datas,
    hiddenimports=ndviewer_light_imports + [
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'ndv',
        'ndv.views._vispy._array_canvas',
        'ndv.models._data_wrapper',
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
        'OpenGL.platform.win32',
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
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    strip=False,
    upx=True,
    console=True,  # TODO: set to False for release builds
    # icon='ndviewer_light.ico',  # TODO: add .ico file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ndviewer_light',
)
