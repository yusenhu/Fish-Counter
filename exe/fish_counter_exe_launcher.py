"""Standalone launcher entrypoint for packaging a one-click Fish Counter EXE.

This file is intentionally separate from the core development CLI script.
Package this file with PyInstaller to create a distributable EXE.
"""

import argparse
import platform
from pathlib import Path

import cv2
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


def find_available_camera(max_index: int = 10) -> int:
    """Return first available camera index."""
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return idx
    raise RuntimeError(f"No camera detected in indices 0..{max_index}.")


def run_live_counter(model_path: str, conf: float, camera_index: int, device: str):
    """Run model tracking/counting overlay in a live camera window."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index: {camera_index}")

    model = YOLO(model_path)
    model = model.to(device)

    total_tracks = set()

    print("[INFO] Live counting started. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=conf, tracker="bytetrack.yaml")
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy
            ids = results[0].boxes.id

            boxes_list = boxes.cpu().tolist() if hasattr(boxes, "cpu") else boxes.tolist()
            ids_list = ids.cpu().tolist() if hasattr(ids, "cpu") else ids.tolist()

            for box, track_id_val in zip(boxes_list, ids_list):
                track_id = int(track_id_val)
                total_tracks.add(track_id)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Fish Count (tracks): {len(total_tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Fish Counter Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="One-click fish counter launcher")
    parser.add_argument("--model", default="training_data/best.pt", help="Path to trained model file (best.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index; -1 means auto detect")
    parser.add_argument("--max-camera-index", type=int, default=10, help="Max index when auto-scanning cameras")
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|0")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Put your trained model at training_data/best.pt "
            "or launch with --model <path>."
        )

    camera_index = args.camera if args.camera >= 0 else find_available_camera(args.max_camera_index)
    device = resolve_device(args.device)

    run_live_counter(str(model_path), args.conf, camera_index, device)


if __name__ == "__main__":
    main()
