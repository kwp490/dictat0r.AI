# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for dictat0r.AI — Granite + Cohere engines (transformers/torch).

Build: pyinstaller dictator.spec
Output: dist/dictator/dictator.exe (onedir)
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# Collect PortAudio DLL from sounddevice
binaries = collect_dynamic_libs('sounddevice')

# Collect transformers data files (tokenizer configs, etc.)
datas = []
try:
    datas += collect_data_files('transformers')
except Exception:
    pass

a = Analysis(
    ['dictator/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [
        ('dictator/assets', 'dictator/assets'),
    ],
    hiddenimports=[
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'sounddevice',
        'soundfile',
        '_soundfile_data',
        'numpy',
        'keyboard',
        'pynvml',
        'transformers',
        'torch',
        'torchaudio',
        'huggingface_hub',
        'sentencepiece',
        'protobuf',
        'tokenizers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # GUI / image libraries not used
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        # Qt submodules not used (only QtWidgets/QtCore/QtGui needed)
        'PySide6.QtQuick',
        'PySide6.QtQml',
        'PySide6.QtPdf',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Strip unnecessary binaries ───────────────────────────────────────────────
import re as _re

_STRIP_PATTERNS = [
    _re.compile(r'Qt6Quick', _re.I),
    _re.compile(r'Qt6Qml', _re.I),
    _re.compile(r'Qt6Pdf', _re.I),
    _re.compile(r'opengl32sw', _re.I),
]

def _should_keep(entry):
    name = entry[0] if isinstance(entry, tuple) else str(entry)
    return not any(p.search(name) for p in _STRIP_PATTERNS)

a.binaries = [b for b in a.binaries if _should_keep(b)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dictator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name='dictator',
)
