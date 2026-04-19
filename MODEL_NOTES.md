# Model Notes (Development & Debug History)

This file keeps technical rationale, change history, and deeper implementation notes.
User-facing quick usage belongs in `README.md`.

## Evolution Summary

### 1) Initial issue: stale model path
Early versions commonly used a fixed model path (`runs/detect/train/weights/best.pt`), which can become stale after multiple training runs (`train2`, `train3`, ...).

### 2) Model auto-selection added
The code now supports selecting newest checkpoint automatically in test/live via `--model auto`, scanning `runs/detect/train*/weights/best.pt` by modification time.

### 3) Directional counting logic added
Counting moved from naive per-detection counting to ID-based line crossing with tracker IDs, with axis modes `auto|horizontal|vertical`.

### 4) Label index collision bug fixed
A real bug existed in label mode when index was computed as `len(existing_images)`. If numbering became non-contiguous, files could be overwritten.

Fix implemented:
- Parse numeric suffix from existing `frame_<n>` files in both image and label folders
- Choose next index as `max(n) + 1`

### 5) Cross-platform training device handling
Training now supports `--device` and resolves safer defaults:
- CUDA GPU when available
- Apple MPS on Apple Silicon
- CPU fallback

Intel Mac explicitly rejects `--device mps` with a clear error.

### 6) NumPy ABI constraint
Because some dependencies may be built against NumPy 1.x, environment is pinned to `numpy<2` to avoid ABI crashes in mixed binary setups.

### 7) NumPy 2.x Compatibility Update (2026)
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

### 8) Dataset Conversion and Training Pipeline (2026)
**Issue**: Original code only supported manual dataset preparation and basic training. Scaling to multiple datasets required automation.

**Features Added**:
- `convert_dataset.py`: Multi-format converter (CSV, JSON, CVAT XML, cropped images)
- `convert_fish_dataset.py`: Auto-detect dataset type and convert
- `merge_datasets.py`: Combine multiple YOLO datasets
- `train.py`: Separate training script with pretrain/finetune/combined modes
- Checkpoint reuse logic to avoid retraining existing models

**Root Cause**: Manual conversion was error-prone and time-consuming for diverse dataset sources.

**Fix Applied**:
- Modular converters with format detection
- Training pipeline supporting transfer learning (pretrain → finetune)
- Automatic best checkpoint detection and reuse

**Impact**: Streamlined workflow from raw datasets to trained models, supporting Kaggle competitions, CVAT annotations, and custom formats.

### 8) FFmpeg Read Attempts Fix for Large Videos (2026)
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

### 9) PowerShell Execution Policy Handling (2026)
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

### 10) Environment Setup Improvements (2026)
**Issue**: Various environment and dependency issues during initial setup.

**Fixes Applied**:
- Updated `requirements.txt` to use `numpy` instead of `numpy<2`
- Added automatic environment variable configuration
- Improved error handling and user guidance
- Added verification commands to README.md

**Impact**: Smoother first-time setup experience.
### 11) GPU Priority and Auto-Dataset Creation (2026)
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
