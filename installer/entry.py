"""Frozen entry point for PyInstaller-built ndviewer_light."""
# MUST be first: in a frozen app, multiprocessing spawn re-launches the
# bundle binary. freeze_support() short-circuits the child relaunch so it
# does NOT re-run main() and open another window. No-op in the parent.
import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import traceback

if getattr(sys, "frozen", False):
    _meipass = sys._MEIPASS
    if sys.platform == "win32":
        os.environ["QT_PLUGIN_PATH"] = os.path.join(
            _meipass, "PyQt5", "Qt5", "plugins"
        )
    elif sys.platform == "darwin":
        # macOS: PyInstaller rewrites @loader_path for bundled dylibs, so we
        # only need to point Qt at its plugin tree. Touching DYLD_LIBRARY_PATH
        # can break framework loading and is stripped by SIP for child procs.
        qt_plugins = os.path.join(_meipass, "PyQt5", "Qt5", "plugins")
        os.environ["QT_PLUGIN_PATH"] = qt_plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            qt_plugins, "platforms"
        )
    else:
        # Linux: set both plugin paths + LD_LIBRARY_PATH for bundled .so files
        qt_plugins = os.path.join(_meipass, "PyQt5", "Qt5", "plugins")
        os.environ["QT_PLUGIN_PATH"] = qt_plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            qt_plugins, "platforms"
        )
        qt_lib = os.path.join(_meipass, "PyQt5", "Qt5", "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_meipass}:{qt_lib}:{existing_ld}"
    os.environ["VISPY_DATA_DIR"] = os.path.join(_meipass, "vispy")
    if sys.platform == "darwin":
        # Inside a .app bundle, sys.executable lives at Contents/MacOS/ which
        # is read-only after Gatekeeper translocation. Log to ~/Library.
        _log_dir = os.path.expanduser("~/Library/Logs/NDViewerLight")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, "crash.log")
    else:
        _log_path = os.path.join(os.path.dirname(sys.executable), "crash.log")

if "--smoke-test" in sys.argv:
    from installer.smoke_test import run
    run()
else:
    try:
        from ndviewer_light.core import main
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if getattr(sys, "frozen", False):
            with open(_log_path, "w") as f:
                f.write(tb)
            print(f"\nCrash log written to: {_log_path}", file=sys.stderr)
        raise
