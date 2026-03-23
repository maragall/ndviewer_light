#!/bin/bash
set -e
cd "$(dirname "$0")"
python -m PyInstaller ndviewer_light_linux.spec --noconfirm --distpath ../dist --workpath ../build --clean
echo "Build complete. Output in dist/ndviewer_light/"
