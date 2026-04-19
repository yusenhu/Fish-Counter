import os
import argparse
import shutil
from pathlib import Path
from typing import Optional
from typing import Optional
from ultralytics import YOLO
import platform


def get_device():
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            # Multi-GPU support
            return "0" if torch.cuda.device_count() == 1 else "0,1"
        elif platform.system() == "Darwin" and platform.processor() == "arm":
            # Apple Silicon
            return "mps"
    except:
        pass
    return "cpu"


def cleanup_weights(save_dir: Path):
    """Keep only best.pt in the weights folder after training."""
    weights_dir = save_dir / "weights"
    if not weights_dir.exists():
        return

    for path in weights_dir.iterdir():
        if path.name != "best.pt":
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                print(f"[WARNING] Could not remove {path}: {e}")


def get_existing_pretrain_checkpoint(pretrain_name: str = "online_fish") -> Optional[Path]:
    """Return the existing best pretrained checkpoint if it exists."""
    path = Path("runs") / "pretrain" / pretrain_name / "weights" / "best.pt"
    return path if path.exists() else None


def pretrain(pretrain_yaml: str, epochs: int = 50, device: Optional[str] = None, force: bool = False):
    """
    Pretrain on large online fish dataset.
    
    Args:
        pretrain_yaml: Path to dataset YAML for online data
        epochs: Number of epochs for pretraining
        force: If True, retrain even if existing best checkpoint already exists
    """
    if device is None:
        device = get_device()

    existing_checkpoint = get_existing_pretrain_checkpoint()
    if existing_checkpoint and not force:
        print(f"[INFO] Existing pretrained checkpoint found: {existing_checkpoint}")
        print("[INFO] Skipping pretraining and reusing the existing best checkpoint.")
        return existing_checkpoint
    if device is None:
        device = get_device()

    existing_checkpoint = get_existing_pretrain_checkpoint()
    if existing_checkpoint and not force:
        print(f"[INFO] Existing pretrained checkpoint found: {existing_checkpoint}")
        print("[INFO] Skipping pretraining and reusing the existing best checkpoint.")
        return existing_checkpoint
    
    print(f"[INFO] Starting pretraining on online dataset...")
    print(f"[INFO] Using device: {device}")
    
    # Load base YOLOv8 model (nano for faster training, or 's'/'m' for better accuracy)
    model = YOLO("yolov8n.pt")  # nano model
    
    # Train on online dataset
    results = model.train(
        data=pretrain_yaml,
        epochs=epochs,
        imgsz=640,
        device=device,
        batch=16,  # Adjust based on GPU memory
        patience=20,
        save=True,
        project="runs/pretrain",
        name="online_fish"
    )

    if results is None:
        raise ValueError("Pretraining failed - no results returned")

    cleanup_weights(results.save_dir)
    
    # Return path to best weights from pretraining
    return results.save_dir / "weights" / "best.pt"


def finetune(pretrain_weights: str, finetune_yaml: str, epochs: int = 25, device: Optional[str] = None):
    """
    Fine-tune pretrained model on your top-view dataset.
    
    Args:
        pretrain_weights: Path to pretrained weights (from pretrain step)
        finetune_yaml: Path to your dataset YAML (top-view data)
        epochs: Number of epochs for fine-tuning
        device: Device ('0', 'mps', 'cpu', or None for auto)
    """
    if device is None:
        device = get_device()
    
    print(f"[INFO] Starting fine-tuning on top-view dataset...")
    print(f"[INFO] Loading pretrained weights: {pretrain_weights}")
    print(f"[INFO] Using device: {device}")
    
    # Load pretrained model
    model = YOLO(pretrain_weights)
    
    # Fine-tune on your top-view data (lower learning rate, fewer epochs)
    results = model.train(
        data=finetune_yaml,
        epochs=epochs,
        imgsz=640,
        device=device,
        batch=16,
        patience=10,
        save=True,
        project="runs/detect",
        name="topview_finetuned"
    )
    if results is None:
        raise ValueError("Finetuning failed - no results returned")
    cleanup_weights(results.save_dir)
    return results.save_dir / "weights" / "best.pt"


def combined_training(pretrain_yaml: str, finetune_yaml: str, 
                     pretrain_epochs: int = 50, finetune_epochs: int = 25, 
                     device: Optional[str] = None):
    """
    Execute full workflow: pretrain then fine-tune.
    
    Usage:
        python train.py --mode combined --pretrain dataset_pretrain.yaml --finetune dataset.yaml
    """
    if device is None:
        device = get_device()
    
    print("=" * 60)
    print("TRANSFER LEARNING WORKFLOW: Pretrain → Fine-tune")
    print("=" * 60)
    
    # Step 1: Pretrain
    print("\n[STEP 1/2] PRETRAINING on online dataset...")
    pretrained_weights = pretrain(pretrain_yaml, pretrain_epochs, device)
    print(f"✓ Pretraining complete. Best weights: {pretrained_weights}\n")
    
    # Step 2: Fine-tune
    print("[STEP 2/2] FINE-TUNING on top-view dataset...")
    final_weights = finetune(str(pretrained_weights), finetune_yaml, finetune_epochs, device)
    print(f"✓ Fine-tuning complete. Final model: {final_weights}\n")
    
    print("=" * 60)
    print(f"✓ TRAINING COMPLETE")
    print(f"  Final model saved to: {final_weights}")
    print(f"  Use with: --model {final_weights}")
    print("=" * 60)
    
    return final_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning: pretrain then fine-tune")
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "combined"],
        default="combined",
        help="Training mode"
    )
    parser.add_argument(
        "--pretrain",
        default="dataset_pretrain.yaml",
        help="Path to pretrain dataset YAML (online fish data)"
    )
    parser.add_argument(
        "--finetune",
        default="dataset.yaml",
        help="Path to fine-tune dataset YAML (your top-view data)"
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Number of pretrain epochs"
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=25,
        help="Number of fine-tune epochs"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: '0', 'mps', 'cpu', or None for auto"
    )
    parser.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Force pretraining even if an existing best checkpoint already exists"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Pretrained weights file to use for finetuning"
    )
    
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        pretrain(args.pretrain, args.pretrain_epochs, args.device, force=args.force_pretrain)
    
    elif args.mode == "finetune":
        if args.checkpoint:
            pretrain_weights = Path(args.checkpoint)
        else:
            pretrain_weights = Path("runs/pretrain/online_fish/weights/best.pt")

        if not pretrain_weights.exists():
            print(f"Error: Pretrain weights not found at {pretrain_weights}")
            print("Run with --mode combined first, or --mode pretrain, or provide --checkpoint")
            exit(1)
        finetune(str(pretrain_weights), args.finetune, args.finetune_epochs, args.device)
    
    elif args.mode == "combined":
        pretrained_weights = pretrain(args.pretrain, args.pretrain_epochs, args.device, force=args.force_pretrain)
        print(f"✓ Pretraining complete. Best weights: {pretrained_weights}\n")
        print("[STEP 2/2] FINE-TUNING on top-view dataset...")
        final_weights = finetune(str(pretrained_weights), args.finetune, args.finetune_epochs, args.device)
        print(f"✓ Fine-tuning complete. Final model: {final_weights}\n")

        print("=" * 60)
        print(f"✓ TRAINING COMPLETE")
        print(f"  Final model saved to: {final_weights}")
        print(f"  Use with: --model {final_weights}")
        print("=" * 60)

