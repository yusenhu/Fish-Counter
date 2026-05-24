"""
Convert Kaggle Fish Dataset (segmentation masks) to YOLO format.

This dataset uses segmentation masks (black/white images) where:
- Black = background
- White = fish

We convert these to bounding boxes for YOLO training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import shutil


def mask_to_bbox(mask_path: str) -> Tuple[int, int, int, int]:

    """
    Convert segmentation mask to bounding box.

    Args:
        mask_path: Path to black/white segmentation mask

    Returns:
        (x1, y1, x2, y2) bounding box coordinates
    """
    # Read mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    # Find white pixels (fish)
    white_pixels = np.where(mask > 127)  # Threshold at 127

    if len(white_pixels[0]) == 0:
        # No fish found, return full image bbox
        h, w = mask.shape
        return 0, 0, w, h

    # Get bounding box
    y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
    x_min, x_max = white_pixels[1].min(), white_pixels[1].max()

    return int(x_min), int(y_min), int(x_max), int(y_max)


def bbox_to_yolo_format(bbox: Tuple[int, int, int, int],
                        img_width: int, img_height: int) -> str:
    """
    Convert bbox (x1, y1, x2, y2) to YOLO format (normalized 0-1).

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        img_width, img_height: Image dimensions

    Returns:
        YOLO format string: "0 x_center y_center width height"
    """
    x1, y1, x2, y2 = bbox

    # Calculate center and dimensions
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Clamp to [0, 1] and ensure minimum size
    x_center = max(0.001, min(0.999, x_center))
    y_center = max(0.001, min(0.999, y_center))
    width = max(0.001, min(0.999, width))
    height = max(0.001, min(0.999, height))

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def convert_kaggle_fish_dataset(dataset_root: str, output_dir: str,
                               max_samples_per_species: Optional[int] = None):
    """
    Convert Kaggle Fish Dataset to YOLO format.

    Args:
        dataset_root: Root directory of Fish_Dataset/Fish_Dataset/
        output_dir: Output YOLO dataset directory
        max_samples_per_species: Limit samples per species (for testing)
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)

    print(f"[INFO] Converting Kaggle Fish Dataset from: {dataset_root}")
    print(f"[INFO] Output directory: {output_dir}")

    # Create YOLO structure
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    species_count = {}

    # Process each species
    species_folders = [f for f in dataset_path.iterdir()
                      if f.is_dir() and not f.name.endswith('.txt')]

    for species_folder in species_folders:
        species_name = species_folder.name
        print(f"[INFO] Processing species: {species_name}")

        # Find image and GT folders
        image_folder = None
        gt_folder = None

        for subfolder in species_folder.iterdir():
            if subfolder.is_dir():
                if "GT" in subfolder.name:
                    gt_folder = subfolder
                else:
                    image_folder = subfolder

        if image_folder is None or gt_folder is None:
            print(f"[WARNING] Skipping {species_name}: missing image or GT folder")
            continue

        # Get all image files
        image_files = list(image_folder.glob("*.png"))
        
        # Optional: limit samples per species for testing
        if max_samples_per_species:
            image_files = image_files[:max_samples_per_species]

        species_count[species_name] = len(image_files)

        for img_file in image_files:
            # Corresponding mask file
            mask_file = gt_folder / img_file.name
            if not mask_file.exists():
                print(f"[WARNING] Mask not found: {mask_file}")
                continue

            # Get image dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"[WARNING] Could not read image: {img_file}")
                continue

            img_height, img_width = img.shape[:2]

            # Convert mask to bbox
            try:
                bbox = mask_to_bbox(str(mask_file))
                yolo_line = bbox_to_yolo_format(bbox, img_width, img_height)
            except Exception as e:
                print(f"[WARNING] Failed to process {img_file}: {e}")
                continue

            # Split into train/val (80/20)
            split = "train" if hash(img_file.name) % 10 < 8 else "val"

            # Copy image
            out_img = output_path / "images" / split / f"{species_name}_{img_file.name}"
            shutil.copy(img_file, out_img)

            # Write label
            label_name = out_img.stem + ".txt"
            out_label = output_path / "labels" / split / label_name
            out_label.write_text(yolo_line)

            total_images += 1

    # Print summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print("Species breakdown:")
    for species, count in species_count.items():
        print(f"  {species}: {count} images")
    print(f"\nOutput: {output_path.absolute()}")
    print("="*60)


def create_dataset_yaml(dataset_dir: str, output_file: str = "dataset_pretrain.yaml"):
    """Create YAML config for the converted dataset."""
    yaml_content = f"""path: {Path(dataset_dir).absolute()}
train: images/train
val: images/val
names:
  0: fish
"""

    with open(output_file, 'w') as f:
        f.write(yaml_content)

    print(f"[INFO] Created dataset config: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Kaggle Fish Dataset (segmentation) to YOLO format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input Fish_Dataset/Fish_Dataset/ directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output YOLO dataset directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per species (for testing)"
    )
    parser.add_argument(
        "--yaml",
        default="dataset_pretrain.yaml",
        help="Output YAML config file"
    )

    args = parser.parse_args()

    # Convert dataset
    convert_kaggle_fish_dataset(
        args.input,
        args.output,
        args.max_samples
    )

    # Create YAML config
    create_dataset_yaml(args.output, args.yaml)