import os
# Set FFmpeg environment variable BEFORE any imports (for 1-2 GB videos)
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "65536"

import argparse
import platform
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import cv2
from ultralytics import YOLO

# ==============================
# ===== DEFAULT PARAMETERS =====
# ==============================

VIDEO_PATH = "test_fish_video.mp4"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 0
FRAME_INTERVAL_SECONDS = 0.5

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


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback for drawing annotation boxes in label mode."""
    global ix, iy, drawing, current_boxes, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if frame_copy is not None:
            temp = frame_copy.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)  # type: ignore
            cv2.imshow("Label Mode", temp)  # type: ignore

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if frame_copy is not None:
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            current_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # type: ignore
            cv2.imshow("Label Mode", frame_copy)  # type: ignore


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
            cv2.imshow("Label Mode", frame_copy)  # type: ignore
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


def find_latest_best_model(default_path: str = "runs/detect/train/weights/best.pt") -> str:
    """Return newest available best.pt across train folders; fallback to YOLO base model."""
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        candidates = []
        for train_dir in runs_dir.glob("train*"):
            best = train_dir / "weights" / "best.pt"
            if best.exists():
                candidates.append(best)

        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            return str(latest)

    # If no trained model exists, fall back to the base YOLO model from ultralytics
    self_warn = """[WARNING] No local trained best model found at runs/detect/train*/weights/best.pt. "
                 "Falling back to 'yolov8n.pt'. Please run training first to generate a local checkpoint."""
    print(self_warn)
    return "yolov8n.pt"


def resolve_model_path(model_arg: str) -> str:
    """Resolve model path from CLI arg (`auto` supported)."""
    if model_arg == "auto":
        path = find_latest_best_model()
        if not Path(path).exists() and path != "yolov8n.pt":
            raise FileNotFoundError(
                f"Auto-selected model path does not exist: {path}. "
                "Run training mode first or specify --model <path>.")
        print(f"[INFO] Auto-selected model: {path}")
        return path

    if not Path(model_arg).exists():
        raise FileNotFoundError(f"Model not found: {model_arg}")

    return model_arg


def resolve_train_device(device_arg: str) -> str:
    """Resolve training device with GPU priority."""
    import torch

    is_mac = platform.system() == "Darwin"
    is_intel_mac = is_mac and platform.machine() == "x86_64"

    if device_arg == "auto":
        # Prioritize GPU when available
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                return "0,1"  # Use multiple GPUs if available
            return "0"  # Use first GPU
        if torch.backends.mps.is_available() and not is_intel_mac:
            return "mps"  # Apple Silicon GPU
        return "cpu"  # Fallback to CPU

    if device_arg == "mps" and is_intel_mac:
        raise RuntimeError(
            "MPS is only supported on Apple Silicon Macs. "
            "Detected Intel Mac; use --device cpu."
        )

    return device_arg


def create_dataset_yaml():
    """Create dataset.yaml if it doesn't exist."""
    import yaml

    dataset_yaml_path = "dataset.yaml"

    if Path(dataset_yaml_path).exists():
        print(f"[INFO] Dataset config already exists: {dataset_yaml_path}")
        return

    # Check if dataset directories exist
    images_dir = Path("dataset/images")
    labels_dir = Path("dataset/labels")

    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created images directory: {images_dir}")

    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created labels directory: {labels_dir}")

    # Create dataset.yaml
    dataset_config = {
        "path": "dataset",
        "train": "images",
        "val": "images",  # Use same images for both train and val for small datasets
        "names": {
            0: "fish"
        }
    }

    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Created dataset config: {dataset_yaml_path}")


def train_mode(resume: bool, epochs: int, imgsz: int, batch: int, device: str):
    """Train model from base checkpoint or resume from latest best model."""
    # Ensure dataset.yaml exists
    create_dataset_yaml()

    if resume:
        resume_model = find_latest_best_model()
        print(f"[INFO] Continuing training from: {resume_model}")
        model = YOLO(resume_model)
    else:
        print("[INFO] Starting NEW training from YOLO base model (yolov8n.pt)...")
        model = YOLO("yolov8n.pt")

    resolved_device = resolve_train_device(device)
    print(f"[INFO] Training device: {resolved_device}")

    model.train(
        data="dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        device=resolved_device,
    )


class DirectionalCounter:
    """
    Count fish once when crossing a center line.
    Axis can be fixed or auto-selected based on track motion.
    """

    def __init__(self, axis_mode: str = "auto", min_track_points: int = 4):
        self.axis_mode = axis_mode  # auto | horizontal | vertical
        self.min_track_points = min_track_points
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.counted_ids = set()
        self.total_count = 0
        self.left_to_right = 0
        self.right_to_left = 0
        self.top_to_bottom = 0
        self.bottom_to_top = 0

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
        h, w = frame_shape[:2]

        start_x, start_y = history[0]
        end_x, end_y = history[-1]

        if axis == "horizontal":
            line_x = w / 2.0
            start_side = "left" if start_x < line_x else "right"
            end_side = "left" if end_x < line_x else "right"

            if start_side != end_side:
                self.counted_ids.add(track_id)
                self.total_count += 1
                if start_side == "left" and end_side == "right":
                    self.left_to_right += 1
                elif start_side == "right" and end_side == "left":
                    self.right_to_left += 1

        else:  # vertical axis
            line_y = h / 2.0
            start_side = "top" if start_y < line_y else "bottom"
            end_side = "top" if end_y < line_y else "bottom"

            if start_side != end_side:
                self.counted_ids.add(track_id)
                self.total_count += 1
                if start_side == "top" and end_side == "bottom":
                    self.top_to_bottom += 1
                elif start_side == "bottom" and end_side == "top":
                    self.bottom_to_top += 1

    def draw_overlay(self, frame):
        axis = self._choose_axis()
        h, w = frame.shape[:2]

        if axis == "horizontal":
            line_x = int(w / 2)
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
            line_y = int(h / 2)
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


def run_detection_stream(capture: cv2.VideoCapture, model_path: str, conf: float, axis_mode: str, window_name: str, device: str = "auto"):
    model = YOLO(model_path)

    # Force GPU usage when available and requested
    if device not in {"cpu", "none"}:
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "0"
            elif torch.backends.mps.is_available() and platform.system() == "Darwin":
                device = "mps"
            else:
                device = "cpu"

        try:
            model = model.to(device)
            print(f"[INFO] Running detection on device: {device}")
        except Exception as e:
            print(f"[WARNING] Cannot move model to {device}: {e} (fallback to cpu)")
            model = model.to("cpu")

    counter = DirectionalCounter(axis_mode=axis_mode)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=conf, tracker="bytetrack.yaml")

        if results and len(results) > 0:
            detection_result = results[0]
            if detection_result.boxes is not None and detection_result.boxes.id is not None:
                boxes = detection_result.boxes.xyxy
                box_ids = detection_result.boxes.id

                # Convert tensors to list, handling both torch tensors and numpy arrays
                if hasattr(boxes, 'cpu'):  # torch tensor
                    boxes_list = boxes.cpu().tolist()  # type: ignore
                else:  # numpy array
                    boxes_list = boxes.tolist()

                if hasattr(box_ids, 'cpu'):  # torch tensor
                    ids_list = [int(v) for v in box_ids.cpu().tolist()]  # type: ignore
                else:  # numpy array
                    ids_list = [int(v) for v in box_ids.tolist()]

                for box, track_id in zip(boxes_list, ids_list):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    counter.update(track_id, (cx, cy), frame.shape)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # type: ignore
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
        cv2.imshow(window_name, frame)  # type: ignore

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def detect_and_count_video(video_path: str, model_path: str, conf: float, axis_mode: str, device: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    run_detection_stream(cap, model_path, conf, axis_mode, "Video Fish Detection", device)


def live_camera_mode(camera_index: int, model_path: str, conf: float, axis_mode: str, device: str):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index: {camera_index}")
    run_detection_stream(cap, model_path, conf, axis_mode, "Live Fish Detection", device)


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

    # train args
    parser.add_argument("--resume", action="store_true", help="Resume training from latest best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, mps, 0, 0,1...")

    # label args
    parser.add_argument("--frame-interval", type=float, default=FRAME_INTERVAL_SECONDS)

    args = parser.parse_args()


    if args.mode == "train":
        train_mode(resume=args.resume, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
    elif args.mode in {"test", "live"}:
        model_path = resolve_model_path(args.model)
        resolved_device = resolve_train_device(args.device)

        if args.mode == "test":
            detect_and_count_video(args.video, model_path, args.conf, args.axis, resolved_device)
        else:
            live_camera_mode(args.camera, model_path, args.conf, args.axis, resolved_device)
    elif args.mode == "label":
        label_mode(args.video, args.frame_interval)
