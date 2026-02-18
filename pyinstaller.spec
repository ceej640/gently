# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Gently Electron backend.

One-directory mode — fast startup, no temp extraction.
Run from repo root:  pyinstaller --clean --noconfirm pyinstaller.spec
"""

import os
import sys
from pathlib import Path

block_cipher = None
repo_root = Path(SPECPATH)

# ---- Data files (non-Python assets) ----
datas = [
    # Web templates & static assets
    (str(repo_root / "gently" / "visualization" / "web" / "templates"), "gently/visualization/web/templates"),
    (str(repo_root / "gently" / "visualization" / "web" / "static"), "gently/visualization/web/static"),
    # Config
    (str(repo_root / "config"), "config"),
    # Organism modules (contain data files: YAML, JSON, etc.)
    (str(repo_root / "gently" / "organisms"), "gently/organisms"),
    # Hardware module
    (str(repo_root / "gently" / "hardware"), "gently/hardware"),
    # Stage example images + metadata for VLM perception
    (str(repo_root / "gently" / "examples"), "gently/examples"),
]

# ---- Hidden imports (not detected by static analysis) ----
hiddenimports = [
    # uvicorn internals
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # Web framework
    "jinja2",
    "jinja2.ext",
    # Anthropic SDK
    "anthropic",
    # Data processing
    "yaml",
    "scipy",
    "scipy.ndimage",
    "skimage",
    "skimage.measure",
    "skimage.morphology",
    "skimage.filters",
    # Standard library modules sometimes missed
    "sqlite3",
    "multiprocessing",
    # FastAPI / Starlette
    "starlette.responses",
    "starlette.websockets",
    "starlette.staticfiles",
    "starlette.templating",
    # Gently subpackages
    "gently.organisms.celegans",
    "gently.hardware.dispim",
]

# ---- Excludes (heavy deps not needed for agent-only mode) ----
excludes = [
    "pymmcore",
    "bluesky",
    "ophyd",
    "napari",
    "torch",
    "segment_anything",
    "matplotlib",
    "tkinter",
    "_tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "IPython",
    "notebook",
    "jupyterlab",
]

a = Analysis(
    [str(repo_root / "gently" / "electron_entry.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    name="electron_entry",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Needs stdout for READY protocol
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="electron_entry",
)
