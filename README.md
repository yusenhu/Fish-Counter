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

## Repository Layout

The repo is now split into two top-level folders:

- `development/` → source code and scripts used by developers (training, data conversion, testing, CLI workflows)
- `exe/` → EXE packaging and runtime launcher files for non-developer deployment

---

## Quick Start (Development)

### 1) Get files
Clone or download this repository. Required files:

- `development/fish_counter.py` (inference and labeling)
- `development/convert_dataset.py` (dataset conversion utilities)
- `development/convert_fish_dataset.py` (auto-detect dataset converter)
- `development/merge_datasets.py` (merge multiple YOLO datasets)
- `development/train.py` (training script with pretrain/finetune modes)
- `development/requirements.txt`
- `dataset.yaml` (for your top-view data)
- `dataset_pretrain.yaml` (for online pretraining data)
- `dataset/images` and `dataset/labels` (your labeled data)

### 2) Install dependencies

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r development/requirements.txt
```

#### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r development/requirements.txt
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
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
python development/convert_fish_dataset.py --input path/to/your/dataset --output datasets/processed/your_dataset
```

Supports: CSV/JSON annotations, CVAT XML, cropped images, Kaggle segmentation masks.

### Manual converters

#### CSV/JSON to YOLO
```bash
python development/convert_dataset.py csv path/to/annotations.csv path/to/images datasets/processed/output
```

#### CVAT XML to YOLO
```bash
python development/convert_dataset.py xml path/to/annotations.xml path/to/images datasets/processed/output
```

#### Cropped images to YOLO
```bash
python development/convert_dataset.py cropped path/to/cropped_images datasets/processed/output
```

### Merge datasets

```bash
python development/merge_datasets.py datasets/processed/dataset1 datasets/processed/dataset2 --output datasets/processed/merged
```

Creates `fish_dataset_all_processed.yaml` for training.

---

## Training

Train YOLOv8 models with pretraining on online fish data and finetuning on your top-view data.

### Pretrain only (online data)

```bash
python development/train.py --mode pretrain --pretrain dataset_pretrain.yaml --pretrain-epochs 50
```

### Finetune only (your data)

```bash
python development/train.py --mode finetune --finetune dataset.yaml --checkpoint runs/pretrain/online_fish/weights/best.pt
```

### Combined pretrain + finetune

```bash
python development/train.py --mode combined --pretrain dataset_pretrain.yaml --finetune dataset.yaml
```

---

## Inference (Development CLI)

### Label (annotate frames)

```bash
python development/fish_counter.py --mode label --video test_fish_video.mp4 --frame-interval 0.5
```

### Test (video)

```bash
python development/fish_counter.py --mode test --video test_fish_video.mp4 --model auto --axis auto --device auto
```

### Live camera

```bash
python development/fish_counter.py --mode live --camera 0 --model auto --axis auto --device auto
```

---

## EXE Workflow (Non-Developer Use)

### Files you need

- `exe/fish_counter_exe_launcher.py`
- `exe/build_windows_exe.ps1`
- Trained model at `training_data/best.pt`

### Build the EXE (Windows)

From PowerShell:

```powershell
Set-Location .\exe
.\build_windows_exe.ps1
```

Output:

- `exe/dist/FishCounter.exe`

### Run the EXE

Place `FishCounter.exe` where you can also provide a model file (default expected path `training_data/best.pt`) or launch with custom arguments.

Equivalent launcher command (for testing without packaging):

```powershell
python .\exe\fish_counter_exe_launcher.py --model training_data/best.pt --camera -1 --device auto
```

---
