# Model Notes (Development & Debug History)

This file keeps technical rationale, change history, and deeper implementation notes.
User-facing quick usage belongs in `README.md`.

## Repository Organization Update (2026-05-24)

Repository structure is standardized into two folders:
- `development/`: development scripts and model training/inference code
- `exe/`: Windows executable launcher and packaging script

Migration details:
- Moved `fish_counter.py`, training, conversion, merge scripts, and `requirements.txt` into `development/`
- Moved EXE launcher and build script into `exe/`
- Updated build script to set working directory to its script folder before running PyInstaller

This separation keeps deployment artifacts and developer workflows independent, reducing accidental edits during field deployment prep.

## Evolution Summary

### 1) Initial issue: stale model path
Early versions commonly used a fixed model path (`runs/detect/train/weights/best.pt`), which can become stale after multiple training runs (`train2`, `train3`, ...).

### 2) Model auto-selection added
The code now supports selecting newest checkpoint automatically in test/live via `--model auto`, scanning `runs/detect/train*/weights/best.pt` by modification time.

### 3) Directional counting logic added
Counting moved from naive per-detection counting to ID-based line crossing with tracker IDs, with axis modes `auto|horizontal|vertical`.

### 4) Region-based counting mode added (2026-06-03)
**Issue**: Center-line crossing worked when fish tracks clearly moved from one side of a line to the other, but it was less aligned with the newer field setup where fish should be counted when they pass through a visible hose/transfer region.

**Root Cause**:
- The previous logic treated the count boundary as a single horizontal or vertical line.
- Fish that moved inside the target passage area without a clean line-crossing track could be missed or counted later than desired.
- Directional subtotals were not the main operational metric for the newer setup; entering the passage region is the key event.

**Fix Applied**:
- Added `region` to the supported `--axis` choices in both the development CLI and the EXE launcher for backward-compatible argument parsing.
- In the code, `region` is stored in `axis_mode` and returned by `_choose_axis()`, but conceptually it is a separate counting mode rather than a physical movement axis.
- Updated `DirectionalCounter` so `region` mode counts each tracked fish ID once when the detection center first enters the counting rectangle.
- The explicit region uses the full frame width and a vertical band one-third of the frame height, centered at 5/8 of the frame height.
- The overlay now draws the region rectangle and displays `Region mode | Enter count` instead of directional line-crossing subtotals.

**Impact**: Counting can now be configured around a passage area instead of a line, which better matches the current fish-transfer counting method while preserving the older line-crossing modes for comparison and fallback.

### 5) Label index collision bug fixed
A real bug existed in label mode when index was computed as `len(existing_images)`. If numbering became non-contiguous, files could be overwritten.

Fix implemented:
- Parse numeric suffix from existing `frame_<n>` files in both image and label folders
- Choose next index as `max(n) + 1`

### 6) Cross-platform training device handling
Training now supports `--device` and resolves safer defaults:
- CUDA GPU when available
- Apple MPS on Apple Silicon
- CPU fallback

Intel Mac explicitly rejects `--device mps` with a clear error.

### 7) NumPy ABI constraint
Because some dependencies may be built against NumPy 1.x, environment is pinned to `numpy<2` to avoid ABI crashes in mixed binary setups.

### 8) NumPy 2.x Compatibility Update (2026)
**Issue**: NumPy 2.0 introduced breaking changes to C API. Original code had runtime check that raised error on NumPy >= 2.

**Root Cause**: 
- NumPy 2.x is backward compatible at Python level
- Modern packages (opencv-python, ultralytics) support NumPy 2.x
- Pre-built wheels for NumPy < 2 unavailable on Python 3.14+

**Fix Applied**:
- Removed `numpy<2` constraint from `requirements.txt`
- Removed runtime NumPy version check in `fish_counter.py`
- Updated to use NumPy 2.4.3 (current version)
- Verified all dependencies work correctly

**Impact**: Code now works with latest NumPy without version conflicts.

### 9) Dataset Conversion and Training Pipeline (2026)
**Issue**: Original code only supported manual dataset preparation and basic training. Scaling to multiple datasets required automation.

**Features Added**:
- `convert_annotation_dataset.py`: Multi-format converter (CSV, JSON, CVAT XML, cropped images)
- `auto_convert_dataset.py`: Auto-detect dataset type and convert
- `merge_datasets.py`: Combine multiple YOLO datasets
- `train.py`: Separate training script with pretrain/finetune/combined modes
- Checkpoint reuse logic to avoid retraining existing models

**Root Cause**: Manual conversion was error-prone and time-consuming for diverse dataset sources.

**Fix Applied**:
- Modular converters with format detection
- Training pipeline supporting transfer learning (pretrain → finetune)
- Automatic best checkpoint detection and reuse

**Impact**: Streamlined workflow from raw datasets to trained models, supporting Kaggle competitions, CVAT annotations, and custom formats.

### 10) FFmpeg Read Attempts Fix for Large Videos (2026)
**Issue**: OpenCV FFmpeg backend warning: "grabFrame packet read max attempts exceeded" for 1-2 GB videos with multiple streams.

**Root Cause**:
- Default FFmpeg read attempts = 4096
- Large/complex videos need more attempts to read packets
- Environment variable `OPENCV_FFMPEG_READ_ATTEMPTS` must be set before cv2 import

**Fix Applied**:
- Set `OPENCV_FFMPEG_READ_ATTEMPTS = "65536"` at script start (before imports)
- Tested with 16384, 32768, 65536 - 65536 eliminates warning for 1-2 GB videos
- Added troubleshooting section to README.md

**Impact**: Labeling mode works without FFmpeg warnings on large video files.

### 11) PowerShell Execution Policy Handling (2026)
**Issue**: PowerShell blocks `.ps1` scripts by default, causing venv activation failures.

**Root Cause**:
- PowerShell execution policy set to Restricted/Default
- `.ps1` activation scripts blocked for security
- `.bat` scripts work because they use different security model

**Fix Applied**:
- Documented both activation methods in README.md
- Added troubleshooting section for execution policy
- Recommended `.bat` file as primary method (always works)

**Impact**: Virtual environment setup works reliably on Windows.

### 12) Environment Setup Improvements (2026)
**Issue**: Various environment and dependency issues during initial setup.

**Fixes Applied**:
- Updated `requirements.txt` to use `numpy` instead of `numpy<2`
- Added automatic environment variable configuration
- Improved error handling and user guidance
- Added verification commands to README.md

**Impact**: Smoother first-time setup experience.

### 13) GPU Priority and Auto-Dataset Creation (2026)
**Issue**: Training failed when `dataset.yaml` missing, and GPU detection wasn't optimal.

**Fixes Applied**:
- Enhanced device resolution to prioritize GPU (multiple GPUs if available)
- Added automatic `dataset.yaml` creation with proper directory structure
- Auto-creates `dataset/images/` and `dataset/labels/` directories
- Uses standard YOLO dataset format with fish class

**Impact**: Training works out-of-the-box, optimal GPU utilization.
## Technical Implementation Notes

### Model switching semantics
- Train without `--resume`: starts from `yolov8n.pt`
- Train with `--resume`: loads latest trained checkpoint
- Test/live with `--model auto`: loads latest checkpoint

### Counting mode semantics
The runtime argument is still named `--axis` in both scripts, so `region` is currently selected through the same option as the line-crossing axes. Internally, `DirectionalCounter.axis_mode` can be `auto`, `horizontal`, `vertical`, or `region`; when it is `region`, the code bypasses line-crossing logic and uses rectangle-entry counting instead.

- `--axis auto`: chooses horizontal or vertical line-crossing based on observed track movement
- `--axis horizontal`: counts each tracker ID once after crossing the center vertical line, with `L->R` and `R->L` subtotals
- `--axis vertical`: counts each tracker ID once after crossing the center horizontal line, with `T->B` and `B->T` subtotals
- `--axis region`: uses the existing `--axis` selector for compatibility, but counts each tracker ID once when its center first enters the configured counting region; this is the newer pass-through-region workflow, not a directional axis

### Environment Configuration
- `OPENCV_FFMPEG_READ_ATTEMPTS = "65536"` set automatically for large video compatibility
- NumPy 2.x compatible (no version constraints)
- Virtual environment activation supports both PowerShell (.ps1) and batch (.bat) methods
- **GPU Priority**: Auto mode detects and uses multiple GPUs when available (`0,1`), single GPU (`0`), MPS, or CPU

### Dataset Auto-Creation
- `dataset.yaml` created automatically if missing during training
- Includes proper directory structure creation (`dataset/images/`, `dataset/labels/`)
- Standard YOLO format with fish class (id: 0)

### Dataset maintenance notes
After new labels are added:
- Ensure `dataset.yaml` paths still match folder structure
- Ensure class mapping (`names`) stays correct
- Re-run training and test with latest model

### Suggested future improvements
- Add explicit run metadata logging (dataset version, commit hash, metrics)
- Add evaluation script to compare newly trained model vs previous best
- Add video preprocessing option for problematic FFmpeg streams
- Add automatic environment variable detection based on video file size
- Add cross-platform video format validation before processing


## EXE File Development History

### A) Standalone launcher split from development CLI
- Added dedicated `fish_counter_exe_launcher.py` as a separate runtime entrypoint for the packaged EXE
- Kept launcher intentionally isolated from the main development script `fish_counter.py`
- Purpose: stable, minimal entrypoint for PyInstaller packaging and live camera runtime usage

### B) Directional counting, region counting, and live camera controls
- Added directional crossing count logic into the launcher
- Draws a center line and counts fish crossing it in `auto`, `horizontal`, or `vertical` modes
- Added `region` as an `--axis` choice for compatibility, even though it behaves as a rectangle-entry counting mode rather than a directional axis
- Tracks directions for line modes: `L->R`, `R->L`, `T->B`, `B->T`
- Added runtime key controls:
  - `s` = start/resume counting
  - `p` = pause counting
  - `r` = reset counts
  - `q` = quit

### C) Windows packaging automation
- Added `build_windows_exe.ps1` to install packaging dependencies and build the EXE
- Uses PyInstaller with:
  - `pyinstaller --onefile --windowed --name FishCounter --distpath dist fish_counter_exe_launcher.py`
- Builds output to `dist\FishCounter.exe`

### D) Build/runtime path reliability
- Updated launcher to resolve `training_data/best.pt` relative to the executable directory if the default path is missing
- Allows the EXE to find the model when run from `dist\FishCounter.exe`
- Updated the build script to create `dist\training_data` and recommend placing `best.pt` there

### Notes
- The EXE now supports live camera selection and explicit camera scanning
- The EXE supports `--axis region` for the current pass-through-region counting method; this is implemented through the existing axis argument but behaves as a separate region-counting mode
- The runtime experience and packaging path handling are both improved
