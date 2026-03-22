# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for ndviewer_light."""

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

vispy_datas = collect_data_files('vispy')

a = Analysis(
    ['entry.py'],
    pathex=['..'],
    binaries=[],
    datas=vispy_datas,
    hiddenimports=[
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
        'zarr',
        'zarr.storage',
        'OpenGL',
        'OpenGL.GL',
        'OpenGL.platform.win32',
        'numpy.core._methods',
        'numpy.lib.format',
        'xml.etree.ElementTree',
        'importlib.metadata',
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
    console=False,
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
