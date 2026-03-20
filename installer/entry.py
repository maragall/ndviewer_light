"""Frozen entry point for PyInstaller-built ndviewer_light."""
import os
import sys

if getattr(sys, "frozen", False):
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        sys._MEIPASS, "PyQt5", "Qt5", "plugins"
    )
    os.environ["VISPY_DATA_DIR"] = os.path.join(sys._MEIPASS, "vispy")

if "--smoke-test" in sys.argv:
    from installer.smoke_test import run
    run()
else:
    from ndviewer_light.core import main
    main()
