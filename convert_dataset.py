"""
Convert online fish datasets to YOLO format.

Supports:
1. CSV/JSON annotations with bounding boxes
2. Cropped images (auto-generate centered annotations)
3. Directory structures from common sources
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from typing import List, Tuple


def create_yolo_dataset_structure(output_dir: str, train_split: float = 0.8):
    """Create YOLO dataset directory structure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        (Path(output_dir) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "labels" / split).mkdir(parents=True, exist_ok=True)


def bbox_to_yolo_format(bbox: Tuple[float, float, float, float], 
                        img_width: int, img_height: int) -> str:
    """
    Convert bbox (x1, y1, x2, y2) to YOLO format (class x_center y_center width height).
    
    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        img_width, img_height: Image dimensions
    
    Returns:
        YOLO format string (normalized 0-1)
    """
    x1, y1, x2, y2 = bbox
    
    # Center and dimensions
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def convert_csv_annotations(csv_file: str, images_dir: str, output_dir: str):
    """
    Convert CSV annotations to YOLO format.
    
    CSV format expected:
        image_name, x1, y1, x2, y2, [width, height]
    
    Example:
        fish_001.jpg, 10, 20, 150, 180, 640, 480
    """
    print(f"[INFO] Converting CSV annotations: {csv_file}")
    
    create_yolo_dataset_structure(output_dir)
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if exists
        
        for row in reader:
            img_name = row[0].strip()
            x1, y1, x2, y2 = map(int, row[1:5])
            
            # Image dimensions (if provided)
            if len(row) > 5:
                img_width, img_height = int(row[5]), int(row[6])
            else:
                # Load from actual image
                import cv2
                img = cv2.imread(os.path.join(images_dir, img_name))
                if img is None:
                    print(f"[WARNING] Could not load image: {img_name}")
                    continue
                img_height, img_width = img.shape[:2]
            
            # Convert to YOLO format
            yolo_line = bbox_to_yolo_format((x1, y1, x2, y2), img_width, img_height)
            
            # Copy image and create label
            img_path = os.path.join(images_dir, img_name)
            label_name = Path(img_name).stem + ".txt"
            
            # Split into train/val (80/20)
            split = "train" if hash(img_name) % 10 < 8 else "val"
            
            # Copy image
            out_img = os.path.join(output_dir, "images", split, img_name)
            shutil.copy(img_path, out_img)
            
            # Write label
            out_label = os.path.join(output_dir, "labels", split, label_name)
            with open(out_label, 'w') as lf:
                lf.write(yolo_line)
    
    print(f"✓ Conversion complete. Output: {output_dir}")


def convert_json_annotations(json_file: str, images_dir: str, output_dir: str):
    """
    Convert JSON annotations to YOLO format.
    
    JSON format expected (COCO format):
    {
        "images": [{"id": 1, "file_name": "fish_001.jpg", "width": 640, "height": 480}],
        "annotations": [{"image_id": 1, "bbox": [x, y, w, h]}]
    }
    """
    print(f"[INFO] Converting JSON annotations: {json_file}")
    
    create_yolo_dataset_structure(output_dir)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Build image metadata
    images_meta = {img['id']: img for img in data.get('images', [])}
    
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in images_meta:
            continue
        
        img_meta = images_meta[img_id]
        img_name = img_meta['file_name']
        img_width = img_meta['width']
        img_height = img_meta['height']
        
        # COCO format: bbox is [x, y, width, height]
        x, y, w, h = ann['bbox']
        bbox = (int(x), int(y), int(x + w), int(y + h))
        
        yolo_line = bbox_to_yolo_format(bbox, img_width, img_height)
        
        # Split and copy
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
        
        label_name = Path(img_name).stem + ".txt"
        split = "train" if hash(img_name) % 10 < 8 else "val"
        
        out_img = os.path.join(output_dir, "images", split, img_name)
        shutil.copy(img_path, out_img)
        
        out_label = os.path.join(output_dir, "labels", split, label_name)
        with open(out_label, 'w') as lf:
            lf.write(yolo_line)
    
    print(f"✓ Conversion complete. Output: {output_dir}")


def convert_cvat_xml_annotations(xml_file: str, images_dir: str, output_dir: str):
    """Convert CVAT XML annotations to YOLO format."""
    print(f"[INFO] Converting CVAT XML annotations: {xml_file}")

    create_yolo_dataset_structure(output_dir)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # CVAT XML stores image entries with nested box annotations
    for image in root.findall("image"):
        img_name = image.get("name")
        if img_name is None:
            continue
        img_width = int(float(image.get("width", "0")))
        img_height = int(float(image.get("height", "0")))
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found for annotation: {img_path}")
            continue

        yolo_lines = []
        for box in image.findall("box"):
            x1 = float(box.get("xtl", "0"))
            y1 = float(box.get("ytl", "0"))
            x2 = float(box.get("xbr", "0"))
            y2 = float(box.get("ybr", "0"))
            yolo_lines.append(bbox_to_yolo_format((x1, y1, x2, y2), img_width, img_height))

        if not yolo_lines:
            continue

        split = "train" if hash(img_name) % 10 < 8 else "val"
        out_img = os.path.join(output_dir, "images", split, os.path.basename(img_name))
        shutil.copy(img_path, out_img)

        label_name = Path(img_name).stem + ".txt"
        out_label = os.path.join(output_dir, "labels", split, label_name)
        with open(out_label, 'w') as lf:
            lf.writelines(yolo_lines)

    print(f"✓ Conversion complete. Output: {output_dir}")


def convert_cropped_images(cropped_images_dir: str, output_dir: str, 
                          create_centered_bbox: bool = True):
    """
    Convert cropped fish images to YOLO format.
    
    Assumes: cropped images are already fish-centered.
    If create_centered_bbox=True, creates bounding box covering ~80% of image.
    
    Args:
        cropped_images_dir: Directory containing only cropped fish images
        output_dir: Output YOLO format directory
        create_centered_bbox: If True, place bbox at center covering 80% of image
    """
    print(f"[INFO] Converting cropped images: {cropped_images_dir}")
    
    create_yolo_dataset_structure(output_dir)
    
    cropped_dir = Path(cropped_images_dir)
    image_files = list(cropped_dir.glob("*.[jpJP]*[gG]")) + \
                  list(cropped_dir.glob("*.[pP][nN][gG]"))
    
    for img_path in image_files:
        if create_centered_bbox:
            # Assume cropped image = fish is mostly centered
            # Create bbox covering 80% of image (leaving margins)
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARNING] Could not load image: {img_path}")
                continue
            h, w = img.shape[:2]
            
            # 80% coverage from center
            margin_x = w * 0.1
            margin_y = h * 0.1
            bbox = (int(margin_x), int(margin_y), 
                   int(w - margin_x), int(h - margin_y))
            yolo_line = bbox_to_yolo_format(bbox, w, h)
        else:
            # Fish fills entire image
            yolo_line = "0 0.5 0.5 0.95 0.95\n"  # 95% coverage
        
        # Split and copy
        split = "train" if hash(img_path.name) % 10 < 8 else "val"
        
        out_img = Path(output_dir) / "images" / split / img_path.name
        shutil.copy(img_path, out_img)
        
        label_name = img_path.stem + ".txt"
        out_label = Path(output_dir) / "labels" / split / label_name
        out_label.write_text(yolo_line)
    
    print(f"✓ Converted {len(image_files)} images. Output: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert fish datasets to YOLO format")
    parser.add_argument("--csv", help="Path to CSV annotation file")
    parser.add_argument("--json", help="Path to JSON annotation file (COCO format)")
    parser.add_argument("--xml", help="Path to CVAT XML annotation file")
    parser.add_argument("--cropped", help="Directory of cropped fish images")
    parser.add_argument("--images", required=True, help="Directory containing original images")
    parser.add_argument("--output", required=True, help="Output YOLO dataset directory")
    
    args = parser.parse_args()
    
    if args.csv:
        convert_csv_annotations(args.csv, args.images, args.output)
    elif args.json:
        convert_json_annotations(args.json, args.images, args.output)
    elif args.xml:
        convert_cvat_xml_annotations(args.xml, args.images, args.output)
    elif args.cropped:
        convert_cropped_images(args.cropped, args.output)
    else:
        print("Please specify --csv, --json, --xml, or --cropped")
