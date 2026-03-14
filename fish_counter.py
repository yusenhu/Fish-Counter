import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

if int(np.__version__.split(".")[0]) >= 2:
    raise RuntimeError(
        "Detected NumPy >= 2, but this environment has native modules built against NumPy 1.x. "
        "Please install a compatible version: `python -m pip install 'numpy<2' --upgrade --force-reinstall`."
    )

import cv2
from ultralytics import YOLO

# ==============================
# ===== DEFAULT PARAMETERS =====
# ==============================

VIDEO_PATH = "test_fish_video.mp4"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 0
FRAME_INTERVAL_SECONDS = 0.5
LINE_POSITION_RATIO = 0.5
ZONE_MIN_RATIO = 0.4
ZONE_MAX_RATIO = 0.6

# ------------------------------
# Label mode state
# ------------------------------
drawing = False
ix, iy = -1, -1
current_boxes = []
frame_copy = None


def get_next_frame_index(images_dir: str = "dataset/images", labels_dir: str = "dataset/labels") -> int:
    """Return next safe frame index based on max numeric suffix in image/label files."""
    pattern = re.compile(r"^frame_(\d+)\.(jpg|jpeg|png|txt)$", re.IGNORECASE)
    max_idx = -1

    for folder in (images_dir, labels_dir):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

    return max_idx + 1


def find_latest_best_model(default_path: str = "runs/detect/train/weights/best.pt") -> str:
    """Return newest available best.pt across train folders; fallback to default path."""
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        return default_path

    candidates = []
    for train_dir in runs_dir.glob("train*"):
        best = train_dir / "weights" / "best.pt"
        if best.exists():
            candidates.append(best)

    if not candidates:
        return default_path

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)

def draw_rectangle(event, x, y, flags, param):
    """Mouse callback for drawing annotation boxes in label mode."""
    global ix, iy, drawing, current_boxes, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = frame_copy.copy()
        cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Label Mode", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        current_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Label Mode", frame_copy)


def label_mode(video_path: str, frame_interval_seconds: float):
    """Step through video and save labeled frames in YOLO format."""
    global frame_copy, current_boxes

    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/labels", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(fps * frame_interval_seconds))

    frame_index = get_next_frame_index("dataset/images", "dataset/labels")

    print("[INFO] Label Mode Started")
    print("Mouse drag to draw box")
    print("Press 's' to save frame + labels")
    print("Press 'n' for next frame")
    print("Press 'r' to reset boxes")
    print("Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1
        frame_copy = frame.copy()
        current_boxes = []

        cv2.namedWindow("Label Mode")
        cv2.setMouseCallback("Label Mode", draw_rectangle)

        while True:
            cv2.imshow("Label Mode", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                img_name = f"frame_{frame_index}.jpg"
                img_path = os.path.join("dataset/images", img_name)
                cv2.imwrite(img_path, frame)

                label_name = f"frame_{frame_index}.txt"
                label_path = os.path.join("dataset/labels", label_name)
                h, w, _ = frame.shape

                with open(label_path, "w", encoding="utf-8") as f:
                    for box in current_boxes:
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"0 {cx} {cy} {bw} {bh}\n")

                print(f"[INFO] Saved frame {frame_index}")
                frame_index += 1
                break

            if key == ord("n"):
                print("[INFO] Skipped frame")
                break

            if key == ord("r"):
                frame_copy = frame.copy()
                current_boxes = []
                print("[INFO] Reset boxes")

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()


def resolve_model_path(model_arg: str) -> str:
    """Resolve model path from CLI arg (`auto` supported)."""
    if model_arg == "auto":
        path = find_latest_best_model()
        print(f"[INFO] Auto-selected model: {path}")
        return path

    if not Path(model_arg).exists():
        raise FileNotFoundError(f"Model not found: {model_arg}")

    return model_arg


def resolve_train_device(device_arg: str) -> str:
    """Resolve training device with Intel-Mac-safe defaults."""
    import torch

    is_mac = platform.system() == "Darwin"
    is_intel_mac = is_mac and platform.machine() == "x86_64"

    if device_arg == "auto":
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available() and not is_intel_mac:
            return "mps"
        return "cpu"

    if device_arg == "mps" and is_intel_mac:
        raise RuntimeError(
            "MPS is only supported on Apple Silicon Macs. "
            "Detected Intel Mac; use --device cpu."
        )

    return device_arg


def train_mode(resume: bool, epochs: int, imgsz: int, batch: int, device: str):
    """Train model from base checkpoint or resume from latest best model."""
    if resume:
        resume_model = find_latest_best_model()
        print(f"[INFO] Continuing training from: {resume_model}")
        model = YOLO(resume_model)
    else:
        print("[INFO] Starting NEW training from YOLO base model (yolov8n.pt)...")
        model = YOLO("yolov8n.pt")

    resolved_device = resolve_train_device(device)
    print(f"[INFO] Training device: {resolved_device}")

    results = model.train(
        data="dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        device=resolved_device,
    )
    save_training_metadata(results)


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def _dataset_yaml_fingerprint(path: str = "dataset.yaml") -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    content = p.read_bytes()
    import hashlib

    return hashlib.sha256(content).hexdigest()[:12]


def save_training_metadata(train_results):
    save_dir = Path(getattr(train_results, "save_dir", "runs/detect"))
    metrics = dict(getattr(train_results, "results_dict", {}) or {})
    payload = {
        "commit": _git_commit_hash(),
        "dataset_yaml_fingerprint": _dataset_yaml_fingerprint(),
        "save_dir": str(save_dir),
        "metrics": metrics,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "run_metadata.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote run metadata: {out}")


class DirectionalCounter:
    """
    Count fish once when crossing a center line.
    Axis can be fixed or auto-selected based on track motion.
    """

    def __init__(
        self,
        axis_mode: str = "auto",
        min_track_points: int = 4,
        line_position_ratio: float = LINE_POSITION_RATIO,
        zone_min_ratio: Optional[float] = None,
        zone_max_ratio: Optional[float] = None,
    ):
        self.axis_mode = axis_mode  # auto | horizontal | vertical
        self.min_track_points = min_track_points
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.counted_ids = set()
        self.total_count = 0
        self.left_to_right = 0
        self.right_to_left = 0
        self.top_to_bottom = 0
        self.bottom_to_top = 0
        self.line_position_ratio = line_position_ratio
        self.zone_min_ratio = zone_min_ratio
        self.zone_max_ratio = zone_max_ratio

    def _sides_for_axis(self, axis: str, start: Tuple[float, float], end: Tuple[float, float], frame_shape):
        h, w = frame_shape[:2]
        use_zone = self.zone_min_ratio is not None and self.zone_max_ratio is not None

        if axis == "horizontal":
            start_v, end_v = start[0], end[0]
            size = w
        else:
            start_v, end_v = start[1], end[1]
            size = h

        if use_zone:
            zone_min = self.zone_min_ratio * size
            zone_max = self.zone_max_ratio * size

            def classify(val):
                if val < zone_min:
                    return "before"
                if val > zone_max:
                    return "after"
                return "inside"

            return classify(start_v), classify(end_v)

        line_pos = self.line_position_ratio * size
        return ("before" if start_v < line_pos else "after"), ("before" if end_v < line_pos else "after")

    def _choose_axis(self) -> str:
        if self.axis_mode in {"horizontal", "vertical"}:
            return self.axis_mode

        total_dx, total_dy = 0.0, 0.0
        used_tracks = 0

        for history in self.track_history.values():
            if len(history) < self.min_track_points:
                continue
            start_x, start_y = history[0]
            end_x, end_y = history[-1]
            total_dx += abs(end_x - start_x)
            total_dy += abs(end_y - start_y)
            used_tracks += 1

        if used_tracks == 0:
            return "horizontal"

        return "horizontal" if total_dx >= total_dy else "vertical"

    def update(self, track_id: int, center: Tuple[float, float], frame_shape: Tuple[int, int, int]):
        self.track_history[track_id].append(center)

        history = self.track_history[track_id]
        if len(history) < self.min_track_points:
            return

        if track_id in self.counted_ids:
            return

        axis = self._choose_axis()
        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        start_side, end_side = self._sides_for_axis(axis, (start_x, start_y), (end_x, end_y), frame_shape)

        if start_side == "inside" or end_side == "inside":
            return

        if start_side == end_side:
            return

        if axis == "horizontal":
            self.counted_ids.add(track_id)
            self.total_count += 1
            if start_side == "before" and end_side == "after":
                self.left_to_right += 1
            elif start_side == "after" and end_side == "before":
                self.right_to_left += 1

        else:  # vertical axis
            self.counted_ids.add(track_id)
            self.total_count += 1
            if start_side == "before" and end_side == "after":
                self.top_to_bottom += 1
            elif start_side == "after" and end_side == "before":
                self.bottom_to_top += 1

    def draw_overlay(self, frame):
        axis = self._choose_axis()
        h, w = frame.shape[:2]

        if axis == "horizontal":
            if self.zone_min_ratio is not None and self.zone_max_ratio is not None:
                zone_min = int(w * self.zone_min_ratio)
                zone_max = int(w * self.zone_max_ratio)
                cv2.rectangle(frame, (zone_min, 0), (zone_max, h), (255, 255, 0), 2)
            else:
                line_x = int(w * self.line_position_ratio)
                cv2.line(frame, (line_x, 0), (line_x, h), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Axis: horizontal | L->R: {self.left_to_right}  R->L: {self.right_to_left}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
        else:
            if self.zone_min_ratio is not None and self.zone_max_ratio is not None:
                zone_min = int(h * self.zone_min_ratio)
                zone_max = int(h * self.zone_max_ratio)
                cv2.rectangle(frame, (0, zone_min), (w, zone_max), (255, 255, 0), 2)
            else:
                line_y = int(h * self.line_position_ratio)
                cv2.line(frame, (0, line_y), (w, line_y), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Axis: vertical | T->B: {self.top_to_bottom}  B->T: {self.bottom_to_top}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Total Fish Count: {self.total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )


def build_tracker_config(track_high_thresh: float, track_low_thresh: float, new_track_thresh: float, track_buffer: int):
    cfg = (
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {track_high_thresh}\n"
        f"track_low_thresh: {track_low_thresh}\n"
        f"new_track_thresh: {new_track_thresh}\n"
        f"track_buffer: {track_buffer}\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(cfg)
    tmp.flush()
    tmp.close()
    return tmp.name


def run_detection_stream(capture: cv2.VideoCapture, model_path: str, conf: float, axis_mode: str, window_name: str, line_position_ratio: float, zone_min_ratio: Optional[float], zone_max_ratio: Optional[float], tracker_cfg_path: str):
    model = YOLO(model_path)
    counter = DirectionalCounter(
        axis_mode=axis_mode,
        line_position_ratio=line_position_ratio,
        zone_min_ratio=zone_min_ratio,
        zone_max_ratio=zone_max_ratio,
    )

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=conf, tracker=tracker_cfg_path)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            ids = [int(v) for v in results[0].boxes.id.cpu().tolist()]

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                counter.update(track_id, (cx, cy), frame.shape)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        counter.draw_overlay(frame)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def detect_and_count_video(video_path: str, model_path: str, conf: float, axis_mode: str, line_position_ratio: float, zone_min_ratio: Optional[float], zone_max_ratio: Optional[float], tracker_cfg_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    run_detection_stream(cap, model_path, conf, axis_mode, "Video Fish Detection", line_position_ratio, zone_min_ratio, zone_max_ratio, tracker_cfg_path)


def live_camera_mode(camera_index: int, model_path: str, conf: float, axis_mode: str, line_position_ratio: float, zone_min_ratio: Optional[float], zone_max_ratio: Optional[float], tracker_cfg_path: str):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index: {camera_index}")
    run_detection_stream(cap, model_path, conf, axis_mode, "Live Fish Detection", line_position_ratio, zone_min_ratio, zone_max_ratio, tracker_cfg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish detection, tracking, and directional counting")
    parser.add_argument("--mode", choices=["train", "test", "live", "label"], required=True)

    # shared runtime args
    parser.add_argument("--model", default="auto", help='Model path or "auto" for newest runs/detect/train*/weights/best.pt')
    parser.add_argument("--video", default=VIDEO_PATH, help="Video path for test/label modes")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index for live mode")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument(
        "--axis",
        choices=["auto", "horizontal", "vertical"],
        default="auto",
        help="Counting axis: horizontal(left<->right), vertical(top<->bottom), or auto",
    )
    parser.add_argument("--line-pos", type=float, default=LINE_POSITION_RATIO, help="Line position ratio in frame [0-1]")
    parser.add_argument("--zone-min", type=float, default=None, help="Optional zone start ratio [0-1]")
    parser.add_argument("--zone-max", type=float, default=None, help="Optional zone end ratio [0-1]")

    # train args
    parser.add_argument("--resume", action="store_true", help="Resume training from latest best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, mps, 0, 0,1...")

    # label args
    parser.add_argument("--frame-interval", type=float, default=FRAME_INTERVAL_SECONDS)
    parser.add_argument("--track-high-thresh", type=float, default=0.5)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)

    args = parser.parse_args()


    if args.zone_min is not None and args.zone_max is not None and args.zone_min >= args.zone_max:
        raise ValueError("--zone-min must be < --zone-max")

    if args.mode == "train":
        train_mode(resume=args.resume, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
    elif args.mode == "test":
        tracker_cfg_path = build_tracker_config(
            args.track_high_thresh,
            args.track_low_thresh,
            args.new_track_thresh,
            args.track_buffer,
        )
        model_path = resolve_model_path(args.model)
        detect_and_count_video(args.video, model_path, args.conf, args.axis, args.line_pos, args.zone_min, args.zone_max, tracker_cfg_path)
    elif args.mode == "live":
        tracker_cfg_path = build_tracker_config(
            args.track_high_thresh,
            args.track_low_thresh,
            args.new_track_thresh,
            args.track_buffer,
        )
        model_path = resolve_model_path(args.model)
        live_camera_mode(args.camera, model_path, args.conf, args.axis, args.line_pos, args.zone_min, args.zone_max, tracker_cfg_path)
    elif args.mode == "label":
        label_mode(args.video, args.frame_interval)
