"""Frozen entry point for PyInstaller-built ndviewer_light."""
import os
import sys
import traceback

if getattr(sys, "frozen", False):
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        sys._MEIPASS, "PyQt5", "Qt5", "plugins"
    )
    os.environ["VISPY_DATA_DIR"] = os.path.join(sys._MEIPASS, "vispy")
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
            input("Press Enter to close...")
        raise
