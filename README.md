# ndviewer_light

Minimal NDV-based viewer for viewing 5-D acquisitions.

## Installation

### Standalone (pip)

Requires Python 3.10+.

```bash
git clone https://github.com/Cephla-Lab/ndviewer_light.git
cd ndviewer_light
pip3 install .
```

> On Ubuntu 24.04+, add `--break-system-packages` or use a virtual environment.

This installs an `ndviewer-light` command you can run from anywhere.

#### Create a desktop shortcut (optional)

```bash
python3 create_shortcut.py
```

- **macOS:** Creates an app in `~/Applications/NDViewer Light.app`. Drag it to your Dock if you like.
- **Windows:** Creates a shortcut on your Desktop.
- **Linux:** Creates a `.desktop` entry in `~/.local/share/applications/`.

### Conda environment

```bash
conda env create -f environment.yml
conda activate ndviewer_light
```

### As a library

```bash
pip3 install /path/to/ndviewer_light
```

## Usage

### Command line

```bash
ndviewer-light                      # open with file dialog
ndviewer-light /path/to/dataset     # open a specific dataset
```

### As a library

```python
from ndviewer_light import LightweightViewer, LightweightMainWindow

# Embed viewer widget in your own application
viewer = LightweightViewer("/path/to/dataset")

# Or use the standalone window
window = LightweightMainWindow("/path/to/dataset")
window.show()
```
