# Fish Counter System

Automated, non-invasive fish counting for UC Davis conservation operations.

## Problem Statement

UC Davis teams may transfer thousands to tens of thousands of sensitive fish in a short period. Manual hand counting is labor-intensive and may increase handling stress. This project provides a computer-vision workflow to detect, track, and count fish during transfer.

## Objective

Build a practical counting workflow for fish moving through a 4-inch transfer hose that is:

- Accurate at operational flow speeds
- Robust to bubbles/debris and semi-turbid water
- Fast enough for high-throughput transfer events
- Usable in wet, field-like environments

## Sponsor & Contact

- Dennis Cocherell / Florian Mauduit
  Wildlife, Fish, and Conservation Biology Dept., UC Davis
- decocherell@ucdavis.edu / fmauduit@ucdavis.edu

---

## Quick Start

### 1) Get files
Clone or download this repository. Required files:

- `fish_counter.py`
- `requirements.txt`
- `dataset.yaml`
- `dataset/images` and `dataset/labels`

### 2) Install dependencies

#### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

#### Windows + VS Code

1. Open the repo folder in VS Code.
2. Open terminal (PowerShell).
3. Create and activate venv:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Select interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.
5. Install dependencies:

```powershell
pip install -r requirements.txt
```

### 3) Optional verification

```bash
python -c "import numpy, cv2, ultralytics; print('numpy', numpy.__version__)"
```

---

## Dataset Layout

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

When your dataset structure changes (new folders/classes), update `dataset.yaml` accordingly.

---

## Commands

### Label

```bash
python fish_counter.py --mode label --video test_fish_video.mp4 --frame-interval 0.5
```

Controls: `s` save, `n` skip, `r` reset, `q` quit.

### Train (fresh)

```bash
python fish_counter.py --mode train --epochs 50 --imgsz 640 --batch 8 --device auto
```

### Train (continue latest checkpoint)

```bash
python fish_counter.py --mode train --resume --device auto
```

### Test (video)

```bash
python fish_counter.py --mode test --video test_fish_video.mp4 --model auto --axis auto
```

### Live camera

```bash
python fish_counter.py --mode live --camera 0 --model auto --axis auto
```

---

## Model Selection Behavior

- `--model auto` in **test/live** selects the newest `runs/detect/train*/weights/best.pt`.
- `--resume` in **train** starts from the latest trained checkpoint.
- Without `--resume`, training starts from `yolov8n.pt`.

Recommended multi-training cycle:
1. Add labels/images
2. Verify `dataset.yaml`
3. Train with `--resume`
4. Test/live with `--model auto`

---

## Device Notes

- Intel Mac: use `--device cpu`
- Apple Silicon Mac: can use `--device mps`
- Windows/Linux with NVIDIA GPU: use `--device 0`

---

## Troubleshooting

### I merged, but README on GitHub is not the latest

This usually means the merge commit kept the old side of a conflict.

- In conflict resolution, keep the side with your newest edits (often **incoming change** if merging PR branch into an older local branch, but verify by reading the conflict block).
- After resolving, run:

```bash
git add README.md
git commit
```

- Confirm what you are about to push:

```bash
git log --oneline -n 5
git show --name-only --oneline -n 1
```

- Push the branch that contains the merge commit.

### NumPy ABI error
If you see:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Run:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --force-reinstall "numpy<2"
python -m pip install --upgrade --force-reinstall ultralytics opencv-python
```

### Camera not opening

- Try `--camera 1` or `--camera 2`
- Confirm camera permissions for Terminal/VS Code
- Use a webcam/capture device recognized by OS as a standard video device

---

For development/design history and implementation rationale, see `MODEL_NOTES.md`.
