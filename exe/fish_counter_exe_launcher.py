"""Standalone launcher entrypoint for packaging a one-click Fish Counter EXE.

This file is intentionally separate from the core development CLI script.
Package this file with PyInstaller to create a distributable EXE.
"""

import argparse
import platform
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def resolve_device(device_arg: str) -> str:
    """Resolve runtime inference device."""
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
        raise RuntimeError("MPS is only supported on Apple Silicon Macs.")

    return device_arg


def detect_available_cameras(max_index: int = 10) -> List[int]:
    """Scan and return a list of available camera indices."""
    available = []

    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
        cap.release()

    return available


def choose_camera_index(requested_index: int, max_index: int = 10) -> int:
    """Choose the camera index interactively when needed."""
    if requested_index >= 0:
        cap = cv2.VideoCapture(requested_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return requested_index
        cap.release()
        raise RuntimeError(f"Unable to open requested camera index: {requested_index}")

    candidates = detect_available_cameras(max_index)
    if not candidates:
        raise RuntimeError(f"No camera detected in indices 0..{max_index}.")

    if len(candidates) == 1:
        return candidates[0]

    window_name = "Select Camera"
    instructions = [
        "Multiple cameras found.",
        "Press a number key to choose the camera index.",
        "Press 'q' to quit.",
        "",
        "Available indices:",
    ]
    lines = instructions + [f"  {idx}" for idx in candidates]

    canvas = 255 * np.ones((320, 640, 3), dtype="uint8")
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (20, 40 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(window_name, canvas)

    selected_index = candidates[0]
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Camera selection cancelled by user.")

        if 48 <= key <= 57:  # numeric keys 0-9
            candidate = key - 48
            if candidate in candidates:
                selected_index = candidate
                break

    cv2.destroyWindow(window_name)
    return selected_index


class DirectionalCounter:
    """Count fish either by crossing a center line or entering a region."""

    def __init__(self, axis_mode: str = "auto", min_track_points: int = 4, region_frac: float = 0.33):
        self.axis_mode = axis_mode
        self.min_track_points = min_track_points
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.counted_ids = set()
        self.total_count = 0
        self.left_to_right = 0
        self.right_to_left = 0
        self.top_to_bottom = 0
        self.bottom_to_top = 0

        self.region = None
        self.region_frac = float(region_frac)

    def reset(self) -> None:
        self.track_history.clear()
        self.counted_ids.clear()
        self.total_count = 0
        self.left_to_right = 0
        self.right_to_left = 0
        self.top_to_bottom = 0
        self.bottom_to_top = 0
        self.region = None

    def _choose_axis(self) -> str:
        if self.axis_mode in {"horizontal", "vertical", "region"}:
            return self.axis_mode

        total_dx = 0.0
        total_dy = 0.0
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

    def _ensure_region(self, frame_shape: Tuple[int, int, int]):
        if self.region is not None:
            return

        h, w = frame_shape[:2]
        if self.axis_mode == "region":
            rh = h // 3
            x1 = 0
            y_center = (5 * h) // 8
            y1 = int(y_center - rh // 2)
            if y1 < 0:
                y1 = 0
            y2 = y1 + rh
            if y2 > h:
                y2 = h
                y1 = h - rh
            x2 = w
        else:
            rw = int(w * self.region_frac)
            rh = int(h * self.region_frac)
            x1 = (w - rw) // 2
            y1 = (h - rh) // 2
            x2 = x1 + rw
            y2 = y1 + rh

        self.region = (x1, y1, x2, y2)

    def update(self, track_id: int, center: Tuple[float, float], frame_shape: Tuple[int, int, int]) -> None:
        self.track_history[track_id].append(center)

        history = self.track_history[track_id]
        if track_id in self.counted_ids:
            return

        axis = self._choose_axis()

        if axis == "region":
            self._ensure_region(frame_shape)
            x1, y1, x2, y2 = self.region
            cx, cy = center
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                self.counted_ids.add(track_id)
                self.total_count += 1
            return

        if len(history) < self.min_track_points:
            return

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

        else:
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
        elif axis == "vertical":
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
        else:
            self._ensure_region((h, w, 3))
            x1, y1, x2, y2 = self.region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Region mode | Enter count: {self.total_count}",
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


def run_live_counter(model_path: str, conf: float, camera_index: int, device: str, axis_mode: str):
    """Run directional counting overlay in a live camera window."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index: {camera_index}")

    model = YOLO(model_path)
    model = model.to(device)

    counter = DirectionalCounter(axis_mode=axis_mode)
    counting = False

    print("[INFO] Live camera opened on index %s. Press 's' to start, 'p' to pause, 'r' to reset, 'q' to quit." % camera_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if counting:
            results = model.track(frame, persist=True, conf=conf, tracker="bytetrack.yaml")
            if results and len(results) > 0:
                detection = results[0]
                if detection.boxes is not None and detection.boxes.id is not None:
                    boxes = detection.boxes.xyxy
                    ids = detection.boxes.id

                    if hasattr(boxes, "cpu"):
                        boxes_list = boxes.cpu().tolist()
                    else:
                        boxes_list = boxes.tolist()

                    if hasattr(ids, "cpu"):
                        ids_list = [int(v) for v in ids.cpu().tolist()]
                    else:
                        ids_list = [int(v) for v in ids.tolist()]

                    for box, track_id in zip(boxes_list, ids_list):
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

        state_text = "RUNNING" if counting else "PAUSED"
        cv2.putText(
            frame,
            f"State: {state_text}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 205, 50) if counting else (0, 165, 255),
            2,
        )
        cv2.putText(
            frame,
            "s=Start  p=Pause  r=Reset  q=Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Fish Counter Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            counting = True
        elif key == ord("p"):
            counting = False
        elif key == ord("r"):
            counter.reset()

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="One-click fish counter launcher")
    parser.add_argument("--model", default="training_data/best.pt", help="Path to trained model file (best.pt)")
    parser.add_argument("--conf", type=float, default=0.458, help="Detection confidence")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index; -1 means choose from available cameras")
    parser.add_argument("--max-camera-index", type=int, default=10, help="Max index when scanning for cameras")
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|0")
    parser.add_argument(
        "--axis",
        choices=["auto", "horizontal", "vertical", "region"],
        default="auto",
        help="Counting axis for directional count",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        exe_dir = Path(sys.argv[0]).resolve().parent
        fallback_path = exe_dir / args.model
        if fallback_path.exists():
            model_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Put your trained model at training_data/best.pt "
                "next to the executable or launch with --model <path>."
            )

    camera_index = choose_camera_index(args.camera, args.max_camera_index)
    device = resolve_device(args.device)

    run_live_counter(str(model_path), args.conf, camera_index, device, args.axis)


if __name__ == "__main__":
    main()
