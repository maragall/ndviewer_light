#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$REPO_ROOT/NDViewerLight.AppDir"

echo "=== Building AppImage ==="
echo "Repo root: $REPO_ROOT"

# Clean previous AppDir
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/usr"

# Copy PyInstaller output into AppDir
if [ ! -d "$REPO_ROOT/dist/ndviewer_light" ]; then
    echo "ERROR: dist/ndviewer_light/ not found. Run build.sh first."
    exit 1
fi
cp -r "$REPO_ROOT/dist/ndviewer_light/"* "$APP_DIR/usr/"

# Desktop file and icon
cp "$REPO_ROOT/installer/ndviewer_light.desktop" "$APP_DIR/"
cp "$REPO_ROOT/installer/ndviewer_light.png" "$APP_DIR/"
cp "$REPO_ROOT/installer/ndviewer_light.png" "$APP_DIR/.DirIcon"

# AppRun symlink — points to the PyInstaller executable
ln -sf usr/ndviewer_light "$APP_DIR/AppRun"

echo "AppDir created at: $APP_DIR"

# Download appimagetool if not present
TOOL="$REPO_ROOT/appimagetool-x86_64.AppImage"
if [ ! -f "$TOOL" ]; then
    echo "Downloading appimagetool..."
    wget -q "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" -O "$TOOL"
    chmod +x "$TOOL"
fi

# Build AppImage
echo "Running appimagetool..."
ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APP_DIR" "$REPO_ROOT/NDViewerLight-x86_64.AppImage"

echo ""
echo "=== AppImage built: $REPO_ROOT/NDViewerLight-x86_64.AppImage ==="
echo "Run with: chmod +x NDViewerLight-x86_64.AppImage && ./NDViewerLight-x86_64.AppImage"
