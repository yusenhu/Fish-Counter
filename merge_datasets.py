"""
Merge multiple YOLO format datasets into one.

Useful for combining datasets from different online sources before pretraining.
"""

import shutil
from pathlib import Path
from collections import defaultdict


def merge_yolo_datasets(input_dirs, output_dir, verify=True):
    """
    Merge multiple YOLO format datasets.
    
    Args:
        input_dirs: List of input dataset directories (YOLO format)
        output_dir: Output merged dataset directory
        verify: Verify that images and labels match
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(int)
    file_collisions = []
    
    # Merge datasets
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        print(f"[INFO] Merging: {input_dir}")
        
        for split in ["train", "val"]:
            # Copy images
            src_imgs = input_path / "images" / split
            dst_imgs = output_path / "images" / split
            
            if src_imgs.exists():
                for img_file in src_imgs.glob("*"):
                    dst_file = dst_imgs / img_file.name
                    if dst_file.exists():
                        file_collisions.append(f"{split}/images/{img_file.name}")
                    shutil.copy2(img_file, dst_file)
                    stats[f"{split}_images"] += 1
            
            # Copy labels
            src_labels = input_path / "labels" / split
            dst_labels = output_path / "labels" / split
            
            if src_labels.exists():
                for lbl_file in src_labels.glob("*.txt"):
                    dst_file = dst_labels / lbl_file.name
                    if dst_file.exists():
                        file_collisions.append(f"{split}/labels/{lbl_file.name}")
                    shutil.copy2(lbl_file, dst_file)
                    stats[f"{split}_labels"] += 1
    
    # Verify
    if verify:
        train_imgs = set((output_path / "images" / "train").glob("*"))
        train_lbls = set((output_path / "labels" / "train").glob("*.txt"))
        
        if len(train_imgs) != len(train_lbls):
            print(f"[WARNING] Train set mismatch: {len(train_imgs)} images vs {len(train_lbls)} labels")
        
        val_imgs = set((output_path / "images" / "val").glob("*"))
        val_lbls = set((output_path / "labels" / "val").glob("*.txt"))
        
        if len(val_imgs) != len(val_lbls):
            print(f"[WARNING] Val set mismatch: {len(val_imgs)} images vs {len(val_lbls)} labels")
    
    # Print results
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Train images: {stats['train_images']}")
    print(f"Train labels: {stats['train_labels']}")
    print(f"Val images: {stats['val_images']}")
    print(f"Val labels: {stats['val_labels']}")
    
    if file_collisions:
        print(f"\n[WARNING] {len(file_collisions)} filename collisions (overwrites):")
        for collision in file_collisions[:5]:
            print(f"  - {collision}")
        if len(file_collisions) > 5:
            print(f"  ... and {len(file_collisions) - 5} more")
    
    print(f"\nOutput: {output_path.absolute()}")
    print("="*60)
    
    return output_path


def create_dataset_yaml(dataset_dir, output_file="dataset.yaml"):
    """Create YAML config for merged dataset."""
    yaml_content = f"""path: {Path(dataset_dir).absolute()}
train: images/train
val: images/val
names:
  0: fish
"""
    
    with open(output_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"[INFO] Created {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO format fish datasets"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dataset directories"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged dataset directory"
    )
    parser.add_argument(
        "--yaml",
        default=None,
        help="Create YAML config file (optional)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of image/label counts"
    )
    
    args = parser.parse_args()
    
    # Merge datasets
    output_dir = merge_yolo_datasets(
        args.inputs,
        args.output,
        verify=not args.no_verify
    )
    
    # Create YAML if requested
    if args.yaml:
        create_dataset_yaml(output_dir, args.yaml)
