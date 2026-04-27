"""
Patch VisPy's Qt backend to work around QOpenGLWidget FBO corruption
on NVIDIA Blackwell GPUs (RTX PRO 2000, RTX 5000 series, etc.).

Qt5's QOpenGLWidget renders via an internal framebuffer object that gets
corrupted on Blackwell's OpenGL driver. This patch adds support for a
VISPY_USE_LEGACY_QGLWIDGET=1 environment variable that forces the older
QGLWidget (which renders directly to the window surface and works fine).

Usage:
    python patch_vispy_blackwell.py          # Apply patch + set env var
    python patch_vispy_blackwell.py --check  # Check if patch is needed
    python patch_vispy_blackwell.py --revert # Remove patch

The patch is idempotent — safe to run multiple times.
"""

import argparse
import os
import re
import subprocess
import sys


def find_vispy_qt_backend():
    """Locate VisPy's _qt.py backend file."""
    try:
        import vispy
    except ImportError:
        print("ERROR: VisPy is not installed.")
        sys.exit(1)

    qt_path = os.path.join(os.path.dirname(vispy.__file__), "app", "backends", "_qt.py")
    if not os.path.exists(qt_path):
        print(f"ERROR: Could not find VisPy Qt backend at {qt_path}")
        sys.exit(1)

    return qt_path


PATCH_MARKER = "VISPY_USE_LEGACY_QGLWIDGET"

# --- Patch 1: Add env var check to force legacy QGLWidget import ---

PATCH1_ORIGINAL = """\
        if Version(QT_VERSION_STR) >= Version('5.4.0'):
            from PyQt5.QtWidgets import QOpenGLWidget as QGLWidget
            from PyQt5.QtGui import QSurfaceFormat as QGLFormat
            QT5_NEW_API = True
        else:
            from PyQt5.QtOpenGL import QGLWidget, QGLFormat"""

PATCH1_REPLACEMENT = """\
        if os.environ.get('VISPY_USE_LEGACY_QGLWIDGET', '') == '1':
            # Force legacy QGLWidget to work around QOpenGLWidget FBO
            # corruption on NVIDIA Blackwell GPUs (RTX PRO/5000 series)
            from PyQt5.QtOpenGL import QGLWidget, QGLFormat
        elif Version(QT_VERSION_STR) >= Version('5.4.0'):
            from PyQt5.QtWidgets import QOpenGLWidget as QGLWidget
            from PyQt5.QtGui import QSurfaceFormat as QGLFormat
            QT5_NEW_API = True
        else:
            from PyQt5.QtOpenGL import QGLWidget, QGLFormat"""

# --- Patch 2: Pass QGLFormat via constructor for legacy QGLWidget ---

PATCH2_ORIGINAL = """\
            # Qt4 and Qt5 < 5.4.0 - sharing is explicitly requested
            QGLWidget.__init__(self, p.parent, widget, hint)"""

PATCH2_REPLACEMENT = """\
            # Qt4 and Qt5 < 5.4.0 (or legacy QGLWidget forced for Blackwell)
            # sharing is explicitly requested; pass format via constructor
            QGLWidget.__init__(self, glformat, p.parent, widget, hint)"""

# --- Patch 3: Conditionalize setFormat (legacy QGLWidget doesn't have it) ---

PATCH3_ORIGINAL = """\
        self.setFormat(glformat)"""

PATCH3_REPLACEMENT = """\
        if QT5_NEW_API or PYSIDE6_API or PYQT6_API:
            self.setFormat(glformat)"""

# --- Patch 4: Keep auto buffer swap ON for legacy QGLWidget ---

PATCH4_ORIGINAL = """\
        if not QT5_NEW_API and not PYSIDE6_API and not PYQT6_API:
            # to make consistent with other backends
            self.setAutoBufferSwap(False)"""

PATCH4_REPLACEMENT = """\
        if not QT5_NEW_API and not PYSIDE6_API and not PYQT6_API:
            # to make consistent with other backends
            # Keep auto buffer swap ON for legacy QGLWidget on Blackwell GPUs
            if not os.environ.get('VISPY_USE_LEGACY_QGLWIDGET'):
                self.setAutoBufferSwap(False)"""


def is_patched(content):
    return PATCH_MARKER in content


def apply_patch(qt_path):
    with open(qt_path, "r") as f:
        content = f.read()

    if is_patched(content):
        print("Already patched.")
        return True

    original = content

    # Apply patches in order
    content = content.replace(PATCH1_ORIGINAL, PATCH1_REPLACEMENT, 1)
    content = content.replace(PATCH2_ORIGINAL, PATCH2_REPLACEMENT, 1)
    content = content.replace(PATCH3_ORIGINAL, PATCH3_REPLACEMENT, 1)
    content = content.replace(PATCH4_ORIGINAL, PATCH4_REPLACEMENT, 1)

    if content == original:
        print("WARNING: No replacements made. VisPy version may be incompatible with this patch.")
        print("This patch was written for VisPy 0.14.x. Your version may differ.")
        return False

    with open(qt_path, "w") as f:
        f.write(content)

    print(f"Patched: {qt_path}")
    return True


def revert_patch(qt_path):
    with open(qt_path, "r") as f:
        content = f.read()

    if not is_patched(content):
        print("Not patched, nothing to revert.")
        return

    content = content.replace(PATCH1_REPLACEMENT, PATCH1_ORIGINAL, 1)
    content = content.replace(PATCH2_REPLACEMENT, PATCH2_ORIGINAL, 1)
    content = content.replace(PATCH3_REPLACEMENT, PATCH3_ORIGINAL, 1)
    content = content.replace(PATCH4_REPLACEMENT, PATCH4_ORIGINAL, 1)

    with open(qt_path, "w") as f:
        f.write(content)

    print(f"Reverted: {qt_path}")


def set_env_var():
    """Set VISPY_USE_LEGACY_QGLWIDGET=1 permanently (Windows user env)."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, "VISPY_USE_LEGACY_QGLWIDGET", 0, winreg.REG_SZ, "1")
            winreg.CloseKey(key)
            # Broadcast change
            import ctypes
            ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 2, 5000, ctypes.byref(ctypes.c_long()))
            print("Set VISPY_USE_LEGACY_QGLWIDGET=1 in Windows user environment.")
        except Exception as e:
            print(f"Could not set env var automatically: {e}")
            print("Manually set VISPY_USE_LEGACY_QGLWIDGET=1 in your environment.")
    else:
        shell_rc = os.path.expanduser("~/.bashrc")
        line = 'export VISPY_USE_LEGACY_QGLWIDGET=1'
        try:
            with open(shell_rc, "r") as f:
                if PATCH_MARKER in f.read():
                    print(f"Already in {shell_rc}")
                    return
            with open(shell_rc, "a") as f:
                f.write(f"\n# VisPy Blackwell GPU workaround\n{line}\n")
            print(f"Added {line} to {shell_rc}")
            print("Run 'source ~/.bashrc' or restart your shell.")
        except Exception as e:
            print(f"Could not update {shell_rc}: {e}")
            print(f"Manually add: {line}")


def check_needs_patch():
    """Check if this system has a Blackwell GPU that needs the patch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_info = result.stdout.strip()
        print(f"GPU: {gpu_info}")

        blackwell_keywords = ["RTX PRO", "RTX 5", "Blackwell"]
        needs_patch = any(kw.lower() in gpu_info.lower() for kw in blackwell_keywords)

        if needs_patch:
            print("Blackwell GPU detected — patch is recommended.")
        else:
            print("No Blackwell GPU detected — patch likely not needed.")
        return needs_patch
    except FileNotFoundError:
        print("nvidia-smi not found — cannot detect GPU.")
        return None
    except Exception as e:
        print(f"GPU detection failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Patch VisPy for NVIDIA Blackwell GPU compatibility")
    parser.add_argument("--check", action="store_true", help="Check if patch is needed")
    parser.add_argument("--revert", action="store_true", help="Revert the patch")
    args = parser.parse_args()

    qt_path = find_vispy_qt_backend()
    print(f"VisPy Qt backend: {qt_path}")

    if args.check:
        check_needs_patch()
        with open(qt_path) as f:
            print(f"Patch applied: {is_patched(f.read())}")
        return

    if args.revert:
        revert_patch(qt_path)
        return

    # Apply
    check_needs_patch()
    if apply_patch(qt_path):
        set_env_var()
        print("\nDone. Restart any running napari/VisPy applications.")


if __name__ == "__main__":
    main()
