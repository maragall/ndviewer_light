"""Frozen entry point for PyInstaller-built ndviewer_light."""
import os
import sys
import traceback

if getattr(sys, "frozen", False):
    _meipass = sys._MEIPASS
    if sys.platform == "win32":
        os.environ["QT_PLUGIN_PATH"] = os.path.join(
            _meipass, "PyQt5", "Qt5", "plugins"
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
