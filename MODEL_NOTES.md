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

## Technical Implementation Notes

### Model switching semantics
- Train without `--resume`: starts from `yolov8n.pt`
- Train with `--resume`: loads latest trained checkpoint
- Test/live with `--model auto`: loads latest checkpoint

### Dataset maintenance notes
After new labels are added:
- Ensure `dataset.yaml` paths still match folder structure
- Ensure class mapping (`names`) stays correct
- Re-run training and test with latest model

### Suggested future improvements
- Add explicit run metadata logging (dataset version, commit hash, metrics)
- Add evaluation script to compare newly trained model vs previous best
- Add line/zone configuration from CLI instead of fixed center line
- Add tracker config exposure (`bytetrack` thresholds/buffer) through CLI
- Add CI checks and lightweight unit tests for path/model-index helpers
