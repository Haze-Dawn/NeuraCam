"""
FaceCNN v5.1 Fine-Tuning Script

Loads the v5.0 checkpoint, reconstructs BN running stats, re-initializes
detection heads with corrected initialization, and fine-tunes using BCE loss.

Fixes applied from the v5.0 dead-head investigation:
  - BN running stats reconstructed (100 train-mode forward passes)
  - Head obj_pred bias: -4.6 -> -2.5 (7.6x higher initial sigmoid)
  - Head weight init: normal(0.01) -> kaiming_normal() (17.7x larger)
  - Loss: BalancedFocalLoss -> BCEWithLogitsLoss(pos_weight=99.0)
  - Optimizer: shared LR -> differential LR (FPN 2e-3, heads 5e-3, wd=0)
  - Head weight decay: 1e-4 -> 0.0
  - ModelEMA: BN buffer sync fix (applied in train_face_cnn.py)
  - Checkpoint: saves both EMA + model state_dicts

Usage:
  python -m src.training.finetune_v51 \
      --ckpt models/face_cnn_v5_best.pth \
      --data data/wider_face \
      --output models/face_cnn_v5.1_best.pth \
      --epochs 20 --batch-size 16 \
      --freeze-backbone
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def reconstruct_bn_stats(model, loader, device, num_batches=100):
    model.train()
    print(f"Reconstructing BN running stats over {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=num_batches, desc="BN recon")):
            if i >= num_batches:
                break
            patches, *_ = batch
            patches = patches.to(device)
            _ = model(patches)
    model.eval()
    print("BN reconstruction complete.")


def reinit_heads(model):
    heads = [
        model.fpn.head_p2.obj_pred, model.fpn.head_p2.bbox_pred,
        model.fpn.head_p3.obj_pred, model.fpn.head_p3.bbox_pred,
        model.fpn.head_p4.obj_pred, model.fpn.head_p4.bbox_pred,
    ]
    for head in heads:
        nn.init.kaiming_normal_(head.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(head.bias)
        if hasattr(head, 'out_channels') and head.out_channels == 1:
            nn.init.constant_(head.bias, -2.5)
    print("Re-initialized 6 head layers:")
    print("  obj_pred bias: -4.6 -> -2.5")
    print("  all weights: normal(0.01) -> kaiming_normal")


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith('backbone.'):
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Frozen backbone. Trainable: {n_trainable}/{n_total}")


def make_optimizer(model, lr):
    backbone_params = []
    fpn_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        elif name.startswith('fpn.'):
            fpn_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': 1e-4},
        {'params': fpn_params, 'lr': lr * 0.4, 'weight_decay': 1e-5},
        {'params': head_params, 'lr': lr, 'weight_decay': 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    print(f"Optimizer: AdamW | LR: backbone={lr*0.1:.1e}, FPN={lr*0.4:.1e}, heads={lr:.1e}")
    print(f"  Head weight decay: 0.0")
    return optimizer


def train_epoch(model, loader, optimizer, criterion_cls, criterion_bbox,
                device, scaler=None):
    model.train()
    total_loss = 0.0
    total_obj = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    total_bbox = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    total_batches = 0
    grad_norm_sum = 0.0

    levels = ["p2", "p3", "p4"]
    strides = {"p2": 2, "p3": 4, "p4": 8}

    for batch in tqdm(loader, desc="Train v5.1"):
        (patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4) = batch
        patches = patches.to(device)
        targets = {
            "p2": (hm_p2.to(device), bb_p2.to(device)),
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(patches)
            loss = torch.tensor(0.0, device=device)
            for level in levels:
                pred_obj = outputs[f"{level}_obj"]
                pred_bbox = outputs[f"{level}_bbox"]
                t_hm, t_bb = targets[level]
                stride = strides[level]
                pos_mask = (t_hm > 0.5).float()

                obj_l = criterion_cls(pred_obj, t_hm)
                total_obj[level] += obj_l.item()

                bbox_l = torch.tensor(0.0, device=device)
                if pos_mask.sum() > 0:
                    bbox_l = criterion_bbox(pred_bbox, t_bb, pos_mask, stride)
                total_bbox[level] += bbox_l.item() if isinstance(bbox_l, torch.Tensor) else 0
                loss = loss + obj_l + bbox_l

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            total_batches += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_norm_sum += batch_grad_norm
        total_loss += loss.item()
        total_batches += 1

    n = max(total_batches, 1)
    return (total_loss / n,
            {k: v / n for k, v in total_obj.items()},
            {k: v / n for k, v in total_bbox.items()},
            grad_norm_sum / n)


@torch.no_grad()
def validate(model, loader, criterion_cls, device):
    model.eval()
    val_obj = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    tp = {"p2": 0, "p3": 0, "p4": 0}
    fp = {"p2": 0, "p3": 0, "p4": 0}
    fn = {"p2": 0, "p3": 0, "p4": 0}
    n_batches = 0
    levels = ["p2", "p3", "p4"]

    for batch in tqdm(loader, desc="Val v5.1"):
        (patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4) = batch
        patches = patches.to(device)
        targets = {
            "p2": (hm_p2.to(device), bb_p2.to(device)),
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }

        outputs = model(patches)
        for level in levels:
            pred_obj = outputs[f"{level}_obj"]
            t_hm, _ = targets[level]
            val_obj[level] += criterion_cls(pred_obj, t_hm).item()

            pred_probs = torch.sigmoid(pred_obj)
            gt_bin = (t_hm > 0.5).float()
            pred_bin = (pred_probs > 0.5).float()
            tp[level] += (pred_bin * gt_bin).sum().item()
            fp[level] += (pred_bin * (1 - gt_bin)).sum().item()
            fn[level] += ((1 - pred_bin) * gt_bin).sum().item()

        n_batches += 1

    n = max(n_batches, 1)
    results = {}
    for level in levels:
        p = tp[level] / max(tp[level] + fp[level], 1)
        r = tp[level] / max(tp[level] + fn[level], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[level] = {
            "obj_loss": val_obj[level] / n,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="FaceCNN v5.1 Fine-Tuning")
    parser.add_argument("--ckpt", required=True, help="v5.0 checkpoint to fine-tune")
    parser.add_argument("--data", required=True, help="WIDER Face dataset directory")
    parser.add_argument("--output", default="models/face_cnn_v5.1_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--bn-recon-batches", type=int, default=100)
    parser.add_argument("--pos-weight", type=float, default=99.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--validate-interval", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from src.cv.face_detector_cnn import FaceFCNv5
    from src.training.train_face_cnn import (
        ModelEMA, AnchorFreeGIoULoss, WiderFaceFPNDataset,
    )

    # Step 1: Load datasets
    print("\nLoading WIDER Face datasets...")
    train_dataset = WiderFaceFPNDataset(
        args.data, "train", target_h=480, target_w=640, augment=True)
    val_dataset = WiderFaceFPNDataset(
        args.data, "val", target_h=480, target_w=640, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Step 2: Load checkpoint
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
        print("  Using EMA state_dict from checkpoint")
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print("  Using model_state_dict from checkpoint")
    else:
        state_dict = ckpt
        print("  Using bare state_dict")

    model = FaceFCNv5().to(device)
    model.load_state_dict(state_dict, strict=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 3: Reconstruct BN running stats
    reconstruct_bn_stats(model, train_loader, device, args.bn_recon_batches)

    # Step 4: Re-initialize heads
    reinit_heads(model)

    # Step 5: Optionally freeze backbone
    if args.freeze_backbone:
        freeze_backbone(model)

    # Step 6: Loss function
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    criterion_bbox = AnchorFreeGIoULoss()
    print(f"Loss: BCEWithLogitsLoss(pos_weight={args.pos_weight:.1f}) + GIoU")

    # Step 7: Optimizer
    optimizer = make_optimizer(model, args.lr)

    # Step 8: Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-5)")

    # Step 9: EMA
    ema = None
    if args.ema_decay > 0.0:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"EMA: decay={args.ema_decay} (with BN buffer sync)")

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # Step 10: Training loop
    n_batches = max(1, len(train_dataset) // args.batch_size)
    print(f"\nBatches/epoch: {n_batches} | Epochs: {args.epochs}")
    print("=" * 60)

    best_val_f1 = 0.0
    metrics_path = args.output.replace(".pth", "_metrics.csv")
    csv_headers = [
        "epoch", "train_loss",
        "train_obj_p2", "train_obj_p3", "train_obj_p4",
        "train_bbox_p2", "train_bbox_p3", "train_bbox_p4",
        "val_obj_p2", "val_obj_p3", "val_obj_p4",
        "val_f1_p2", "val_f1_p3", "val_f1_p4",
        "lr", "epoch_time_s", "gpu_mem_mb", "grad_norm",
    ]

    metrics_rows = []
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        current_lr = optimizer.param_groups[-1]["lr"]

        train_loss, train_obj, train_bbox, grad_norm = train_epoch(
            model, train_loader, optimizer,
            criterion_cls, criterion_bbox, device, scaler)

        if ema is not None:
            ema.update(model)

        do_val = (epoch % args.validate_interval == 0) or (epoch == args.epochs - 1)
        epoch_time = time.time() - epoch_t0
        gpu_mem = 0
        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats()

        if do_val:
            val_results = validate(model, val_loader, criterion_cls, device)
            avg_f1 = float(np.mean([v["f1"] for v in val_results.values()]))

            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                checkpoint = {
                    'ema_state_dict': ema.state_dict() if ema is not None else model.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_f1': avg_f1,
                    'lr': current_lr,
                }
                torch.save(checkpoint, args.output)
                print(f"  * New best checkpoint (mean F1={avg_f1:.4f})")

            row = [
                epoch + 1, f"{train_loss:.4f}",
                f"{train_obj['p2']:.4f}", f"{train_obj['p3']:.4f}", f"{train_obj['p4']:.4f}",
                f"{train_bbox['p2']:.4f}", f"{train_bbox['p3']:.4f}", f"{train_bbox['p4']:.4f}",
                f"{val_results['p2']['obj_loss']:.4f}", f"{val_results['p3']['obj_loss']:.4f}", f"{val_results['p4']['obj_loss']:.4f}",
                f"{val_results['p2']['f1']:.4f}", f"{val_results['p3']['f1']:.4f}", f"{val_results['p4']['f1']:.4f}",
                f"{current_lr:.8f}", f"{epoch_time:.2f}", f"{gpu_mem:.1f}", f"{grad_norm:.4f}",
            ]
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | Grad: {grad_norm:.2f} | "
                f"F1: P2={val_results['p2']['f1']:.3f} P3={val_results['p3']['f1']:.3f} P4={val_results['p4']['f1']:.3f} | "
                f"LR={current_lr:.1e} | t={epoch_time:.1f}s" +
                (" *" if avg_f1 == best_val_f1 else "")
            )
        else:
            row = [epoch + 1, f"{train_loss:.4f}"] + ["skip"] * 14 + [
                f"{current_lr:.8f}", f"{epoch_time:.2f}", f"{gpu_mem:.1f}", f"{grad_norm:.4f}"]
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | Grad: {grad_norm:.2f} | LR={current_lr:.1e} | t={epoch_time:.1f}s")

        metrics_rows.append(row)

        with open(metrics_path, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(csv_headers)
            w_csv.writerows(metrics_rows)

        scheduler.step()

    print(f"\nTraining complete. Best mean F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {args.output}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
