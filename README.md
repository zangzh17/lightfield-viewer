# Light-Field Volume Viewer

Interactive 3D volume viewer with real-time head-tracking parallax effect. Built for fluorescence microscopy data (NIR-II light-field imaging), but works with any 3D volume.

**Live demo**: [zangzh17.github.io/lightfield-viewer](https://zangzh17.github.io/lightfield-viewer/)

## Features

- **MIP ray marching** — WebGL shader renders maximum intensity projection from any viewing angle
- **Head tracking** — Move your head slightly to see the 3D structure from different perspectives (via webcam + MediaPipe)
- **Multiple datasets** — Switch between volumes via dropdown selector
- **Adjustable display** — Brightness, gamma, threshold, z-scale, zoom controls

## Adding Your Own 3D Data

### Supported input formats

| Format | Shape convention | Notes |
|--------|-----------------|-------|
| `.npy` | `(H, W, D)` | NumPy array, any dtype |
| `.tif` / `.tiff` | `(Z, H, W)` | 3D TIFF stack (auto-transposed to H, W, D) |

### Option A: GUI tool (recommended)

The `prepare_data.py` tool lets you interactively adjust normalization before export.

```bash
pip install numpy matplotlib tifffile

python tools/prepare_data.py
```

1. Place your raw volume files in `raw_data/`
2. Select file, adjust **Clip Lo/Hi %** and **Gamma** — histogram and MIP preview update in real-time
3. Enter a dataset ID (e.g. `my_sample`) and display name
4. Click **Save** — exports to `data/<id>/`

### Option B: Command-line export

```bash
python tools/export_volume.py --npy path/to/volume.npy --serve
```

Options: `--pct-lo`, `--pct-hi` for percentile clipping. `--serve` starts a local server at `http://localhost:8080`.

### Option C: Manual conversion

Any 3D volume can be converted with a few lines of Python:

```python
import numpy as np

# Load your volume as float32, shape (H, W, D)
volume = ...

# Normalize to uint8
pct_lo, pct_hi, gamma = 0.5, 99.8, 1.0
vmin = np.percentile(volume, pct_lo)
vmax = np.percentile(volume, pct_hi)
v = np.clip((volume - vmin) / (vmax - vmin + 1e-10), 0, 1)
v = np.power(v, gamma)
data = (v * 255).astype(np.uint8)

# Save in WebGL layout: (D, H, W) C-order
data_gl = np.ascontiguousarray(data.transpose(2, 0, 1))
data_gl.tofile("data/my_sample/volume.raw")

# Write metadata
import json
with open("data/my_sample/volume_meta.json", "w") as f:
    json.dump({"width": volume.shape[1], "height": volume.shape[0],
               "depth": volume.shape[2]}, f)
```

Then add an entry to `data/datasets.json`:

```json
[
  {"id": "my_sample", "name": "My Sample"}
]
```

## Data format reference

Each dataset lives in `data/<id>/` with two files:

| File | Description |
|------|-------------|
| `volume.raw` | Flat uint8 binary, WebGL layout: `index = x + y*W + z*W*H` |
| `volume_meta.json` | `{"width": W, "height": H, "depth": D}` |

`data/datasets.json` is the index file listing all available datasets.

## Deploying changes

After exporting new data via the GUI tool or manually:

```bash
git add -A
git commit -m "Update volume data"
git push
```

GitHub Pages auto-deploys in ~1-2 minutes.

## Local development

Serve locally with any static HTTP server:

```bash
python -m http.server 8080
# or
python tools/export_volume.py --serve
```

Open `http://localhost:8080` in a browser.

## Tech stack

- [Three.js](https://threejs.org/) — WebGL volume rendering
- [MediaPipe](https://developers.google.com/mediapipe) — Face landmark detection for head tracking
- Custom GLSL MIP ray marching shader (256 steps)
