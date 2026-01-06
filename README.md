# lightweight_ndviewer

minimal ndv-based viewer for viewing 5-D acquisitions

## Installation

### As a library (for use in other projects)

```bash
pip install -e /path/to/ndviewer_light
```

Or with git submodule:

```bash
git submodule add <repo-url> ndviewer_light
pip install -e ./ndviewer_light
```

### Standalone (conda environment)

```bash
conda env create -f environment.yml
conda activate ndviewer_light
```

## Usage

### As a library

```python
from ndviewer_light import LightweightViewer, LightweightMainWindow

# Embed viewer widget in your own application
viewer = LightweightViewer("/path/to/dataset")

# Or use the standalone window
window = LightweightMainWindow("/path/to/dataset")
window.show()
```

### Standalone

```bash
python ndviewer_light.py
```

Or open a specific dataset:

```bash
python ndviewer_light.py /path/to/dataset
```
