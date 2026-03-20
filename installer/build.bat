@echo off
pyinstaller installer/ndviewer_light.spec --noconfirm
echo Build complete. Output in dist/ndviewer_light/
