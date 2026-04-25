# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ndviewer_light (macOS .app bundle).

IMPORTANT: Run from the installer/ directory:
  cd installer && python -m PyInstaller ndviewer_light_mac.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

vispy_datas = collect_data_files('vispy')

ndviewer_light_imports = collect_submodules('ndviewer_light')
ndv_imports = collect_submodules('ndv')
vispy_imports = collect_submodules('vispy')
cmap_imports = collect_submodules('cmap')
psygnal_imports = collect_submodules('psygnal')

a = Analysis(
    ['entry.py'],
    pathex=[os.path.abspath('..')],
    binaries=[],
    datas=vispy_datas + [
        (os.path.join('..', 'ndviewer_light', 'cephla_logo.svg'), 'ndviewer_light'),
    ],
    hiddenimports=ndviewer_light_imports + ndv_imports + vispy_imports + cmap_imports + psygnal_imports + [
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtSvg',
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
        'OpenGL.platform.darwin',
        'numpy.core._methods',
        'numpy.lib.format',
        'xml.etree.ElementTree',
        'importlib.metadata',
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
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ndviewer_light',
)

app = BUNDLE(
    coll,
    name='NDViewerLight.app',
    icon=None,
    bundle_identifier='com.cephla.ndviewer-light',
    info_plist={
        'CFBundleName': 'NDViewer Light',
        'CFBundleDisplayName': 'NDViewer Light',
        'CFBundleExecutable': 'ndviewer_light',
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleVersion': '0.1.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
        'NSPrincipalClass': 'NSApplication',
        'NSRequiresAquaSystemAppearance': False,
    },
)
