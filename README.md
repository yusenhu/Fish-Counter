# Fish Counter System

## Overview
This project develops a non-invasive system to count endangered fish as they are transferred through a 4-inch hose using a fish-friendly pump at UC Davis.

## Motivation
Manual counting causes stress to fish and is not scalable. This system provides an automated, real-time solution using optical sensing and computer vision.

## Main Script
- `fish_counter.py` includes four modes:
  - `label`: label video frames and save YOLO-format annotations.
  - `train`: train YOLO model on your dataset.
  - `test`: run detection + tracking + directional counting on a video.
  - `live`: run detection + tracking + directional counting on a live camera.

## Key Improvements in Current Version
- **Auto model selection**: `--model auto` selects the newest `runs/detect/train*/weights/best.pt`.
- **Directional counting with reduced repeats**:
  - Counts when a tracked fish crosses the center line.
  - Supports `--axis auto` to infer dominant movement direction automatically.
  - Also supports fixed `--axis horizontal` or `--axis vertical`.

---

## macOS Terminal Setup

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install "numpy<2" ultralytics opencv-python
```

> If OpenCV window display has issues on macOS, make sure you run from a normal terminal session (not a headless environment).

---

## Dataset Layout

Expected YOLO dataset structure:

```text
dataset/
  images/
  labels/
dataset.yaml
```

Example `dataset.yaml`:

```yaml
path: dataset
train: images
val: images
names:
  0: fish
```

(For production, use a proper train/val split rather than using the same folder for both.)

---

## Usage

### Label mode
Extract frames from a video at intervals and annotate fish with mouse drag.

```bash
python fish_counter.py --mode label --video test_fish_video.mp4 --frame-interval 0.5
```

Controls in label window:
- `s`: save image + labels
- `n`: skip frame
- `r`: reset boxes
- `q`: quit

### Train mode
Start a new training run from `yolov8n.pt`:

```bash
python fish_counter.py --mode train --epochs 50 --imgsz 640 --batch 8
```

Resume from latest trained `best.pt`:

```bash
python fish_counter.py --mode train --resume
```

### Test mode (video)
Auto-select latest model and auto-detect movement axis:

```bash
python fish_counter.py --mode test --video test_fish_video.mp4 --model auto --axis auto
```

Use a specific model path:

```bash
python fish_counter.py --mode test --video test_fish_video.mp4 --model runs/detect/train7/weights/best.pt --axis horizontal
```

### Live mode (camera)
Use default camera index 0:

```bash
python fish_counter.py --mode live --camera 0 --model auto --axis auto
```

---

## Recommended Test Video for Reliable Counting
For best results, use videos where fish:
1. Enter from one side of the frame,
2. Move clearly across the frame,
3. Exit the frame on the opposite side.

This matches the line-crossing logic and reduces ambiguous counts.

---

## Notes on Accuracy / Duplicate Counts
- Counting is tied to tracker IDs and line crossing.
- Very crowded scenes, severe occlusion, or abrupt motion can still cause ID switches.
- Improve robustness by:
  - Better lighting and contrast,
  - Stable camera placement,
  - Tuning confidence threshold (`--conf`),
  - Using videos with consistent fish flow direction.


---

## Troubleshooting

### NumPy ABI error (NumPy 2.x vs modules built for 1.x)
If you see an error like:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

it means one or more native modules in your env are not yet compatible with NumPy 2.x.

Run these commands in your **same activated venv**:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --force-reinstall "numpy<2"
python -m pip install --upgrade --force-reinstall ultralytics opencv-python
```

Then verify:

```bash
python -c "import numpy, cv2; import ultralytics; print('numpy', numpy.__version__)"
```

If you are on macOS and still see issues, recreate a clean environment:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install "numpy<2" ultralytics opencv-python
```
