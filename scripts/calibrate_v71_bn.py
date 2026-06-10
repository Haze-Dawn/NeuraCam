"""
FaceCNN v7.1 — Post-Training BN Calibration
=============================================
Recomputes BatchNorm running_mean/running_var using a larger batch size
than was used during training. Run AFTER Phase 1 training completes.

Why: Gradient checkpointing + AMP let us train with actual batch=4
(grad accum for effective batch 16). But BN stats see only 4 images
at a time → noisier estimates. This script recalibrates BN stats
with the true batch size you'll use at inference (e.g., batch 16).

Usage:
  python3 scripts/calibrate_v71_bn.py \
    --checkpoint models/face_cnn_v71_best.pth \
    --data data/face/widerface \
    --batch-size 16 \
    --num-batches 500 \
    --output models/face_cnn_v71_best_calibrated.pth
"""

import os, sys, argparse, math, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.training.train_v71 import WiderDataset, CrossDataset
from src.cv.face_detector_v71 import FaceFCNv7_1

warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")


@torch.no_grad()
def calibrate_bn(model, loader, device, num_batches=None):
    """Recompute BN running_mean/running_var by running the model
    in training mode over the dataset.

    PyTorch's BN layers update running_mean and running_var during
    forward() when model.training=True. We set training=True but
    disable gradient computation to get fresh BN statistics at the
    target batch size.

    This replaces the stored running_mean/running_var with ones
    computed at the calibration batch size.
    """
    model.train()
    model.requires_grad_(False)

    # Reset BN running stats to zero before calibration
    # (otherwise they blend old + new estimates)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()

    seen = 0
    pbar = tqdm(loader, desc="Calibrating BN")
    for batch_idx, (imgs, _) in enumerate(pbar):
        if num_batches is not None and batch_idx >= num_batches:
            break
        imgs = imgs.to(device)
        model(imgs)
        seen += imgs.size(0)

        # Report momentum-adjusted progress
        # BN momentum = 0.1 by default → ~10 batches to converge
        pbar.set_postfix(images=seen)

    model.eval()
    print(f"\nBN calibration complete: {seen} images processed.")
    print("Running stats updated for all BatchNorm layers.")
    return model


def main():
    parser = argparse.ArgumentParser(description="V7.1 BN Calibration")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--data", type=str, default="data/face/widerface",
                        help="WIDER Face dataset root")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for BN calibration (default: 16)")
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Number of batches to run (default: full dataset)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: <checkpoint>_calibrated.pth)")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA state dict instead of raw model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = FaceFCNv7_1(obj_bias=-3.0).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("ema_state_dict" if args.ema else "model_state_dict", ckpt)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataset (no augmentation — just raw images for BN calibration)
    dataset = WiderDataset(args.data, "train", target_h=480, target_w=640,
                           stride=4, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False)

    # Calibrate
    model = calibrate_bn(model, loader, device, args.num_batches)

    # Save
    out_path = args.output or args.checkpoint.replace(".pth", "_calibrated.pth")
    ckpt["model_state_dict"] = model.state_dict()
    if "ema_state_dict" in ckpt:
        ckpt["ema_state_dict"] = model.state_dict()
    torch.save(ckpt, out_path)
    print(f"Calibrated checkpoint saved: {out_path}")


if __name__ == "__main__":
    main()
