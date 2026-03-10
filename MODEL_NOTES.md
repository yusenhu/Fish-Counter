# Model Selection and Fish Counting Recommendations

## Does test/live use the latest trained model automatically?

In the current script, test mode and live mode both load a fixed path:

```python
MODEL_PATH = "runs/detect/train/weights/best.pt"
```

That means they **do not automatically pick the newest training run** if Ultralytics creates `train2`, `train3`, etc. They only use whatever file exists at `runs/detect/train/weights/best.pt`.

### What happens when you train repeatedly

- `model.train(...)` without a fixed `name` typically creates incremented run folders (`train`, `train2`, `train3`, ...).
- New `best.pt` files are saved in each run's `weights/` subfolder.
- Your test/live code still points to `runs/detect/train/weights/best.pt`, so it can stay on an older model.

## Better approach

Use one of these options:

1. **Explicit model argument** (`--model runs/detect/train7/weights/best.pt`) for test/live.
2. **Auto-select latest run** by scanning `runs/detect` and picking the most recently modified `weights/best.pt`.
3. **Pin a stable path** after training by copying/symlinking newest `best.pt` to a fixed location like `models/latest.pt`.

## Counting improvements (avoid repeated counting)

Current code uses global `counted_ids` and increments once per tracker ID, which is a good start. To improve robustness:

- Define a **counting line/zone** and only increment when a track crosses in one direction.
- Require a track to be **stable for N frames** before eligible for counting.
- Set tracker parameters (`track_buffer`, matching thresholds) to reduce ID switches.
- Add **TTL cleanup** for very old IDs in long live sessions.
- Optionally maintain **entry/exit counters** if fish can move both directions.

## Data/training workflow suggestions

- Split data into train/val/test and keep test set fixed for fair comparisons.
- Version datasets (`dataset_v1`, `dataset_v2`) and log which version trained each model.
- Save metadata (date, metrics, dataset version, commit hash) with each run.
- Compare new model vs previous best before promoting to `latest.pt`.

## Practical implementation idea

A small helper can select latest weights for test/live:

```python
from pathlib import Path


def get_latest_best(default="runs/detect/train/weights/best.pt"):
    candidates = list(Path("runs/detect").glob("train*/weights/best.pt"))
    if not candidates:
        return default
    return str(max(candidates, key=lambda p: p.stat().st_mtime))
```

Then set `MODEL_PATH = get_latest_best()` or expose it with argparse.
