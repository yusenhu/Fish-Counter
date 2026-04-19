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

- `fish_counter.py` (inference and labeling)
- `convert_dataset.py` (dataset conversion utilities)
- `convert_fish_dataset.py` (auto-detect dataset converter)
- `merge_datasets.py` (merge multiple YOLO datasets)
- `train.py` (training script with pretrain/finetune modes)
- `requirements.txt`
- `dataset.yaml` (for your top-view data)
- `dataset_pretrain.yaml` (for online pretraining data)
- `dataset/images` and `dataset/labels` (your labeled data)

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

## Dataset Preparation

Before training, convert your datasets to YOLO format.

### Auto-detect converter

```bash
python convert_fish_dataset.py --input path/to/your/dataset --output datasets/processed/your_dataset
```

Supports: CSV/JSON annotations, CVAT XML, cropped images, Kaggle segmentation masks.

### Manual converters

#### CSV/JSON to YOLO
```bash
python convert_dataset.py csv path/to/annotations.csv path/to/images datasets/processed/output
```

#### CVAT XML to YOLO
```bash
python convert_dataset.py xml path/to/annotations.xml path/to/images datasets/processed/output
```

#### Cropped images to YOLO
```bash
python convert_dataset.py cropped path/to/cropped_images datasets/processed/output
```

### Merge datasets

```bash
python merge_datasets.py datasets/processed/dataset1 datasets/processed/dataset2 --output datasets/processed/merged
```

Creates `fish_dataset_all_processed.yaml` for training.

---

## Dataset Layout

```text
datasets/processed/your_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
dataset.yaml
```

Example `dataset.yaml`:

```yaml
path: datasets/processed/your_dataset
train: images/train
val: images/val
names:
  0: fish
```

When your dataset structure changes, update the YAML accordingly.

---

## Training

Train YOLOv8 models with pretraining on online fish data and finetuning on your top-view data.

### Pretrain only (online data)

```bash
python train.py --mode pretrain --pretrain dataset_pretrain.yaml --pretrain-epochs 50
```

### Finetune only (your data)

```bash
python train.py --mode finetune --finetune dataset.yaml --checkpoint runs/pretrain/online_fish/weights/best.pt
```

### Combined pretrain + finetune

```bash
python train.py --mode combined --pretrain dataset_pretrain.yaml --finetune dataset.yaml
```

Automatically reuses existing best checkpoints unless `--force-pretrain` is used.

---

## Inference Commands

### Label (annotate frames)

```bash
python fish_counter.py --mode label --video test_fish_video.mp4 --frame-interval 0.5
```

Controls: `s` save, `n` skip, `r` reset, `q` quit.

### Test (video)

```bash
python fish_counter.py --mode test --video test_fish_video.mp4 --model auto --axis auto --device auto
```

### Live camera

```bash
python fish_counter.py --mode live --camera 0 --model auto --axis auto --device auto
```
```bash
python fish_counter.py --mode live --camera 0 --model auto --axis auto
```

---

## Model Selection Behavior

- `--model auto` in **test/live** selects the newest `runs/detect/train*/weights/best.pt` from finetuning.
- Training uses `yolov8n.pt` as base, with optional pretraining.
- Pretraining saves to `runs/pretrain/online_fish*/weights/best.pt`.
- Finetuning saves to `runs/detect/topview_finetuned*/weights/best.pt`.

Recommended workflow:
1. Convert and merge datasets
2. Pretrain on online data
3. Finetune on your top-view data
4. Test/live with `--model auto`

---

## Device Notes

- **Auto mode** prioritizes GPU: uses multiple GPUs if available (`0,1`), single GPU (`0`), Apple Silicon MPS, or CPU fallback
- Intel Mac: use `--device cpu`
- Apple Silicon Mac: can use `--device mps` or `auto`
- Windows/Linux with NVIDIA GPU: use `--device 0` or `auto`
- Multiple GPUs: `--device 0,1` or let `auto` detect

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
python -m pip install --upgrade --force-reinstall ultralytics opencv-python
```

### Training issues
- If pretraining fails, check `dataset_pretrain.yaml` points to valid processed data
- Use `--force-pretrain` to retrain even if best checkpoint exists
- For custom checkpoints, use `--checkpoint path/to/best.pt` in finetune mode

### FFmpeg read attempts warning

If you see:

```text
[ WARN:...] grabFrame packet read max attempts exceeded... OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096)
```

For large videos (1-2 GB+), the code automatically sets this to 65536. If you still get warnings, try:

```powershell
$env:OPENCV_FFMPEG_READ_ATTEMPTS = "131072"
python fish_counter.py --mode label --video your_video.mp4
```

---

For development/design history and implementation rationale, see `MODEL_NOTES.md`.
