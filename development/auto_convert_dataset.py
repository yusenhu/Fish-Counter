"""
Auto-detect fish dataset format and convert to YOLO.

This wrapper can detect common dataset layouts and choose the right converter:
- segmentation masks -> convert_segmentation_mask_dataset.py
- CSV bounding boxes -> convert_annotation_dataset.py
- COCO-style JSON -> convert_annotation_dataset.py
- cropped images -> convert_annotation_dataset.py
- already-YOLO datasets -> copy or create YAML config
"""

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from convert_annotation_dataset import (
    convert_csv_annotations,
    convert_json_annotations,
    convert_cropped_images,
    convert_cvat_xml_annotations,
)
from convert_segmentation_mask_dataset import convert_kaggle_fish_dataset, create_dataset_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


def has_images(path: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_EXTENSIONS for p in path.rglob("*"))


def find_annotation_file(path: Path, extension: str):
    return next(path.rglob(f"*{extension}"), None)


def is_coco_json(file_path: Path) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "images" in data and "annotations" in data
    except Exception:
        return False


def has_yolo_dataset_structure(path: Path) -> bool:
    images_dir = path / "images"
    labels_dir = path / "labels"
    return (
        images_dir.exists()
        and labels_dir.exists()
        and any(images_dir.rglob("*.*"))
        and any(labels_dir.rglob("*.txt"))
    )


def has_mask_dirs(path: Path) -> bool:
    name_matches = ("mask", "gt", "seg")
    for subdir in path.rglob("*"):
        if subdir.is_dir():
            lower_name = subdir.name.lower()
            if any(token in lower_name for token in name_matches):
                if any(file.suffix.lower() == ".png" for file in subdir.iterdir() if file.is_file()):
                    return True
    return False


def is_cvat_xml(file_path: Path) -> bool:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root.tag == "annotations" and any(root.findall("image"))
    except Exception:
        return False


def is_bbox_csv(file_path: Path) -> bool:
    try:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            sample = next(reader, None)

        if not header or not sample or len(sample) < 5:
            return False

        header_str = ",".join(header).lower()
        if any(token in header_str for token in ["x1", "x2", "y1", "y2", "xtl", "xbr", "ytl", "ybr"]):
            return True

        try:
            [float(sample[i]) for i in range(1, 5)]
            return True
        except Exception:
            return False
    except Exception:
        return False


def detect_dataset_type(dataset_root: str) -> str:
    path = Path(dataset_root)

    if not path.exists():
        raise FileNotFoundError(f"Input dataset root not found: {dataset_root}")

    if has_yolo_dataset_structure(path):
        return "yolo"

    xml_file = find_annotation_file(path, ".xml")
    if xml_file and is_cvat_xml(xml_file):
        return "cvat_xml"

    json_file = find_annotation_file(path, ".json")
    if json_file and is_coco_json(json_file):
        return "json"

    csv_file = find_annotation_file(path, ".csv")
    if csv_file and is_bbox_csv(csv_file):
        return "csv"

    if has_mask_dirs(path):
        return "segmentation"

    if has_images(path):
        return "cropped"

    return "unknown"


def copy_yolo_dataset(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ["images", "labels"]:
        src = input_dir / child
        dst = output_dir / child
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-detect fish dataset format and convert to YOLO format"
    )
    parser.add_argument("--input", required=True, help="Input dataset root directory")
    parser.add_argument("--output", required=True, help="Output YOLO dataset directory")
    parser.add_argument(
        "--dataset-type",
        choices=["segmentation", "csv", "json", "cropped", "yolo", "cvat_xml"],
        default=None,
        help="Force dataset type instead of auto-detecting",
    )
    parser.add_argument(
        "--yaml",
        default="dataset_auto.yaml",
        help="Output YAML dataset config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples per species for segmentation datasets",
    )

    args = parser.parse_args()

    dataset_type = args.dataset_type or detect_dataset_type(args.input)
    print(f"[INFO] Detected dataset type: {dataset_type}")

    output_path = Path(args.output)
    input_path = Path(args.input)

    if dataset_type == "segmentation":
        convert_kaggle_fish_dataset(str(input_path), str(output_path), args.max_samples)
    elif dataset_type == "csv":
        csv_file = find_annotation_file(input_path, ".csv")
        if csv_file is None:
            raise FileNotFoundError("CSV annotation file not found in input directory")
        convert_csv_annotations(str(csv_file), str(input_path), str(output_path))
    elif dataset_type == "json":
        json_file = find_annotation_file(input_path, ".json")
        if json_file is None:
            raise FileNotFoundError("JSON annotation file not found in input directory")
        convert_json_annotations(str(json_file), str(input_path), str(output_path))
    elif dataset_type == "cvat_xml":
        xml_file = find_annotation_file(input_path, ".xml")
        if xml_file is None:
            raise FileNotFoundError("XML annotation file not found in input directory")
        convert_cvat_xml_annotations(str(xml_file), str(input_path), str(output_path))
    elif dataset_type == "cropped":
        convert_cropped_images(str(input_path), str(output_path))
    elif dataset_type == "yolo":
        if input_path.resolve() != output_path.resolve():
            copy_yolo_dataset(input_path, output_path)
        else:
            output_path = input_path
        print(f"[INFO] YOLO dataset detected. Copied to {output_path}")
    else:
        raise ValueError(
            "Unable to auto-detect dataset type. Please use --dataset-type to specify "
            "segmentation, csv, json, cropped, or yolo."
        )

    create_dataset_yaml(str(output_path), args.yaml)
    print(f"[INFO] Created YAML config: {args.yaml}")
