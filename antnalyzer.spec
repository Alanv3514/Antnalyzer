# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\Alan\\Documents\\Antnalyzer\\antnalyzer.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\Alan\\Documents\\Antnalyzer\\assets', 'assets/'), ('C:\\Users\\Alan\\Documents\\Antnalyzer\\src\\modules', 'modules/'), ('C:\\Users\\Alan\\Documents\\Antnalyzer\\src\\models_data', 'models_data/'), ('C:\\Users\\Alan\\Documents\\Antnalyzer\\ultralytics', 'ultralytics/'), ('C:\\Users\\Alan\\Documents\\Antnalyzer\\tkinter', 'tkinter')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='antnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
