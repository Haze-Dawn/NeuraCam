"""
FaceCNN v6.0 — Full Retrain From Scratch (rev. May 26, 2026)

Motivation:
  v5.0 produces zero detections. Root causes:
  1. ModelEMA did NOT track BN buffers → saved checkpoint has frozen BN stats
  2. Head bias stuck at -4.6 (FocalLoss trap) → bias never escaped
  3. BalancedFocalLoss + normal(0.01) init → gradient starvation

v6.0 Key Changes vs v5.0:
  Architecture:       Unchanged (FaceFCNv5, 394K params)
  Head bias init:     -2.5 (v5: -4.6)                    → 7.6x higher sigmoid
  Head weight init:   kaiming_normal (v5: normal 0.01)   → 17.7x larger
  Loss function:      BCEWithLogitsLoss(pos_weight=10/25/50 per level)
  Logit clamping:     min=-10, max=10 before BCE (prevents extreme gradients)
  NaN guard:          nan_to_num on loss + grad NaN skip after backward
  Optimizer:          Diff-LR groups (bb=1e-3,fpn=2e-3,heads=5e-3)
  Head weight decay:  0.0 (v5: 1e-4)
  LR schedule:        Smooth cosine, no restart (v5 restart cost +5.6% loss)
  EMA:                BN buffer sync FIXED
  Progressive:        P4 immediate, P3 ep15, P2 ep30 (prevents P2/P3 dead bias)
  Auto batch-size:    GPU VRAM < 5.5GB → batch 16→8 (OOM prevention)
  Resume:             --resume checkpoint.pth (full state restore)
  Mining:             Diagnostic-only by default (--mine-retrain to enable)
  Checkpoints:        Every 5 epochs + best on detection F1
  Diagnostics:        BN gamma, head weights, output stats, per-epoch JSON

  First-run failure (May 26, 2026): Training ran 1 epoch, loss=43.28 (42x too
  high), grad_norm=NaN. Root cause: pos_weight=99/300/500 combined with
  kaiming init created extreme BCE gradients (~99 per cell for confident-
  wrong positives), causing NaN backpropagation. Fixes: reduced pos_weight
  to 10/25/50, added logit clamping [-10,10], added nan_to_num guard.

  Hard-negative mining bug (May 26, 2026): The hard_negative_mining function
  had two critical bugs causing exponential slowdown (13-27 s/it) and system
  instability during mining at epoch 20:
    1. Unbounded candidates list: stored full 480x640x3 FP32 tensors (~3.68MB
       each) for every crop scoring >0.3 confidence. With early-epoch model,
       hundreds of crops qualify, ballooning CPU RAM by 1-3 GB.
    2. DataLoader num_workers=2: OpenCV cv2.imread in forked workers is not
       fork-safe with many OpenCV threading backends (Qt/TBB/GStreamer),
       causing deadlocks and I/O stalls.
    Fixes: (a) bounded min-heap (top-96) replaces unbounded list, capping CPU
    RAM at ~353 MB; (b) num_workers=0 eliminates fork-deadlock risk.
    Full investigation archived at models/archive/mining_bug_investigation/.

Usage:
  python -m src.training.train_v6 \
      --data /path/to/Data \
      --output models/face_cnn_v6_best.pth \
      --epochs 120 --batch-size 16
"""

import os, sys, time, csv, math, argparse, json, warnings
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Recommended: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# before launching to prevent CUDA OOM from memory fragmentation at batch=16.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments"):
    warnings.warn(
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is recommended to "
        "prevent CUDA OOM from memory fragmentation at batch=16 on 6GB GPUs.",
        UserWarning, stacklevel=0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.training.train_face_cnn import (
    ModelEMA, AnchorFreeGIoULoss,
    WiderFaceFPNDataset, WiderFaceFPNMineDataset,
)


def get_optimizer(model, base_lr=1e-3):
    backbone = []; fpn = []; heads = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone.append(param)
        elif name.startswith('fpn.') and 'head' not in name:
            fpn.append(param)
        else:
            heads.append(param)
    return optim.AdamW([
        {'params': backbone, 'lr': base_lr,           'weight_decay': 1e-4},
        {'params': fpn,      'lr': base_lr * 2,       'weight_decay': 1e-5},
        {'params': heads,    'lr': base_lr * 5,       'weight_decay': 0.0},
    ], lr=base_lr, weight_decay=1e-4)


def _apply_warmup(optimizer, cur_step, warmup_steps, base_lr):
    if cur_step < warmup_steps:
        scale = cur_step / max(1, warmup_steps)
        mults = [1.0, 2.0, 5.0]
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = base_lr * mults[i] * scale


@torch.no_grad()
def collect_diagnostics(model):
    diag = {
        'bn_gamma': {},
        'head_weights': {},
        'head_biases': {},
    }
    for name, param in model.named_parameters():
        if 'bn' in name and 'weight' in name:
            key = name.replace('.weight', '')
            p = param.detach().nan_to_num(nan=0.0)
            diag['bn_gamma'][key] = {
                'mean': float(p.mean().item()),
                'std': float(p.std().item()),
                'min': float(p.min().item()),
                'max': float(p.max().item()),
            }
    for name, buf in model.named_buffers():
        if 'running_mean' in name:
            key = name.replace('.running_mean', '')
            entry = diag['bn_gamma'].setdefault(key, {})
            b = buf.detach().nan_to_num(nan=0.0)
            entry['running_mean_std'] = float(b.std().item())
        if 'running_var' in name:
            key = name.replace('.running_var', '')
            entry = diag['bn_gamma'].setdefault(key, {})
            b = buf.detach().nan_to_num(nan=0.0)
            entry['running_var_mean'] = float(b.mean().item())

    for name in ['head_p2.obj_pred', 'head_p3.obj_pred', 'head_p4.obj_pred']:
        w = model.get_parameter(f'{name}.weight').detach().nan_to_num(nan=0.0)
        b = model.get_parameter(f'{name}.bias').detach().nan_to_num(nan=0.0)
        diag['head_weights'][name] = {
            'l2_norm': float(w.norm(2).item()),
            'std': float(w.std().item()),
            'mean': float(w.mean().item()),
        }
        diag['head_biases'][name] = {
            'value': float(b.item()),
            'sigmoid': float(torch.sigmoid(b).item()),
        }

    for name in ['fpn.lat2.weight', 'fpn.lat3.weight', 'fpn.lat4.weight']:
        try:
            w = model.get_parameter(name).detach().nan_to_num(nan=0.0)
            diag['head_weights'][name] = {
                'std': float(w.std().item()),
                'l2_norm': float(w.norm(2).item()),
            }
        except AttributeError:
            pass

    return diag


@torch.no_grad()
def compute_output_stats(model, val_loader, device, num_batches=20):
    model.eval()
    level_stats = {lv: {'max': [], 'min': [], 'std': [],
                         'above_01': [], 'above_03': [], 'above_05': []}
                   for lv in ['p2', 'p3', 'p4']}

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        patches = batch[0].to(device)
        outputs = model(patches)
        for lv in ['p2', 'p3', 'p4']:
            s = torch.sigmoid(outputs[f'{lv}_obj'])
            s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
            level_stats[lv]['max'].append(float(s.max().item()))
            level_stats[lv]['min'].append(float(s.min().item()))
            level_stats[lv]['std'].append(float(s.std().item()))
            level_stats[lv]['above_01'].append(int((s > 0.10).sum().item()))
            level_stats[lv]['above_03'].append(int((s > 0.30).sum().item()))
            level_stats[lv]['above_05'].append(int((s > 0.50).sum().item()))

    summary = {}
    for lv in ['p2', 'p3', 'p4']:
        summary[lv] = {
            'max_mean': float(np.mean(level_stats[lv]['max'])),
            'max_max': float(np.max(level_stats[lv]['max'])),
            'min_mean': float(np.mean(level_stats[lv]['min'])),
            'std_mean': float(np.mean(level_stats[lv]['std'])),
            'cells_above_01_per_image': float(np.mean(level_stats[lv]['above_01'])),
            'cells_above_03_per_image': float(np.mean(level_stats[lv]['above_03'])),
            'cells_above_05_per_image': float(np.mean(level_stats[lv]['above_05'])),
        }
    model.train()
    return summary


def train_epoch(model, loader, optimizer, criterion_cls, criterion_bbox,
                device, scaler=None, warmup_steps=0, base_lr=1e-3,
                step_start=0, p3_obj_active=True, p2_obj_active=True):
    model.train()
    total_loss = 0.0; total_obj = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    total_bbox = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    total_batches = 0; grad_sum = 0.0
    levels = ["p2", "p3", "p4"]
    strides = {"p2": 2, "p3": 4, "p4": 8}

    for batch_idx, batch in enumerate(tqdm(loader, desc="Train v6")):
        cur_step = step_start + batch_idx
        _apply_warmup(optimizer, cur_step, warmup_steps, base_lr)

        (patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4) = batch
        patches = patches.to(device)
        targets = {
            "p2": (hm_p2.to(device), bb_p2.to(device)),
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(patches)
            loss = torch.tensor(0.0, device=device)

            for level in levels:
                pred_obj = torch.clamp(outputs[f"{level}_obj"], min=-10.0, max=10.0)
                pred_bbox = outputs[f"{level}_bbox"]
                t_hm, t_bb = targets[level]
                pos_mask = (t_hm > 0.5).float()

                if (level == 'p3' and not p3_obj_active) or \
                   (level == 'p2' and not p2_obj_active):
                    obj_l = torch.tensor(0.0, device=device)
                else:
                    obj_l = criterion_cls[level](pred_obj, t_hm)
                    obj_l = torch.nan_to_num(obj_l, nan=0.0)
                total_obj[level] += obj_l.item()

                bbox_l = torch.tensor(0.0, device=device)
                if pos_mask.sum() > 0:
                    bbox_l = criterion_bbox(pred_bbox, t_bb, pos_mask,
                                             strides[level])
                    bbox_l = torch.nan_to_num(bbox_l, nan=0.0)
                total_bbox[level] += bbox_l.item() if isinstance(
                    bbox_l, torch.Tensor) else 0.0
                loss = loss + obj_l + bbox_l

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            if scaler is not None:
                scaler.update()
            total_batches += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters())
            if has_nan_grad:
                optimizer.zero_grad()
                scaler.update()
                total_batches += 1
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters())
            if has_nan_grad:
                optimizer.zero_grad()
                total_batches += 1
                continue
            optimizer.step()

        gn = sum(p.grad.data.norm(2).item() ** 2
                 for p in model.parameters() if p.grad is not None) ** 0.5
        grad_sum += gn
        total_loss += loss.item()
        total_batches += 1

    n = max(total_batches, 1)
    return (total_loss / n,
            {k: v / n for k, v in total_obj.items()},
            {k: v / n for k, v in total_bbox.items()},
            grad_sum / n,
            step_start + len(loader))


@torch.no_grad()
def validate(model, loader, criterion_cls, device):
    model.eval()
    val_obj = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    tp = {"p2": 0, "p3": 0, "p4": 0}
    fp = {"p2": 0, "p3": 0, "p4": 0}
    fn = {"p2": 0, "p3": 0, "p4": 0}
    n_batches = 0; levels = ["p2", "p3", "p4"]

    for batch in tqdm(loader, desc="Val v6"):
        (patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4) = batch
        patches = patches.to(device)
        targets = {
            "p2": (hm_p2.to(device), bb_p2.to(device)),
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }
        outputs = model(patches)
        for level in levels:
            pred_obj = torch.clamp(outputs[f"{level}_obj"], min=-10.0, max=10.0)
            t_hm, _ = targets[level]
            obj_val = criterion_cls[level](pred_obj, t_hm).item()
            if not math.isfinite(obj_val):
                obj_val = 0.0
            val_obj[level] += obj_val
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
            "precision": p, "recall": r, "f1": f1,
            "tp": int(tp[level]), "fp": int(fp[level]), "fn": int(fn[level]),
        }
    return results


import heapq as _heapq

@torch.no_grad()
def hard_negative_mining(model, train_root, device, num_images=2000,
                          conf_thresh=0.3, top_k=96, use_val=False):
    """Hard-negative mining with bounded-memory candidate tracking.

    Args:
        use_val: If True, mine from validation set instead of train.
                 Provides fresh negatives the model hasn't seen in training.

    Fixes three bugs discovered during v6.0 training:

      BUG 1 (CPU RAM exhaustion): Unbounded candidates list stored
        full-resolution (480x640x3) FP32 tensors (~3.68 MB each) for
        every crop scoring >0.3 confidence. With a poorly-calibrated early-epoch model,
        hundreds of crops qualify, ballooning the list to 1-3 GB of CPU
        RAM before sort-and-truncate. On an 8 GB system, this triggers
        OOM killer or swap thrashing.
        FIX: Replace unbounded list with a bounded min-heap of size
        top_k (default 96). Each crop's tensor is only stored if its
        confidence beats the heap's current minimum. Memory is bounded to
        top_k * 3.68 MB ≈ 353 MB worst-case.

      BUG 2 (OpenCV fork deadlock): DataLoader num_workers=2 forked
        worker processes calling cv2.imread/cv2.resize. OpenCV builds
        with certain threading backends (e.g., Qt, TBB, or GStreamer)
        are not fork-safe—forked children inherit locks held by the
        parent, causing deadlocks or corrupted image tensors. The
        resulting I/O stalls compound with bug 1 to produce the observed
        13–27 s/it exponential slowdown.
        FIX: Set num_workers=0. The forward pass on 480x640 FPN images
        (model.eval(), torch.no_grad(), batch=8) runs at ~12 it/s even
        with synchronous data loading, completing ~2000 crops in ~20 s.
        The sync loading overhead (~8 ms/img via cv2.imread) is negligible
        compared to the fork-deadlock risk.

      BUG 3 (CUDA VRAM fragmentation): Over 250+ forward passes through
        the FPN (P2: 320×240, P3: 160×120, P4: 80×60 feature maps per
        batch), the CUDA caching allocator fragments the remaining ~1.5 GB
        GPU headroom into hundreds of small free blocks. Each subsequent
        cudaMalloc takes progressively longer to find a contiguous block,
        degrading from 1-3 s/it to 86 s/it by batch 210 (see reports/
        HARD_MINING_GPU_FRAGMENTATION_BUG.md for full timing curves).
        The fragmentation is permanent — subsequent training epochs retain
        the elevated 4.55–4.59 GB VRAM baseline (vs. 4.05 GB normal).
        FIX: (a) Wrap forward pass in torch.no_grad() to eliminate
        autograd graph construction (~50-100 MB per batch); (b) call
        torch.cuda.empty_cache() every 20 batches to reset the caching
        allocator while the free-block list is still compact.

    Architecture note: WiderFaceFPNMineDataset always returns full-frame
    480x640 crops (crop dimensions equal image dimensions), so each
    "crop" is the entire resized WIDER image. The dataset's
    _crop_avoids_faces simply shifts the crop window to avoid GT faces;
    if no shift avoids faces, the top-left 480x640 region is returned.
    """
    model.eval()
    split = "val" if use_val else "train"
    mine_dataset = WiderFaceFPNMineDataset(
        train_root, split, target_h=480, target_w=640,
        samples_per_image=2, max_crop_attempts=20)
    mine_loader = DataLoader(mine_dataset, batch_size=8,
                              shuffle=True, num_workers=0, pin_memory=True)
    candidates_heap = []   # min-heap of (conf, unique_id, tensor)
    candidate_id = 0       # tie-breaker so heap never compares tensors
    total_evaluated = 0
    max_batches = min(num_images // 8, len(mine_loader))

    for batch_idx, batch in enumerate(tqdm(mine_loader, total=max_batches,
                                            desc="Mine HNs")):
        if batch_idx >= max_batches:
            break
        patches = batch[0].to(device)
        with torch.no_grad():
            outputs = model(patches)
        for b in range(patches.size(0)):
            max_conf = 0.0
            for level in ["p2", "p3", "p4"]:
                obj = torch.sigmoid(outputs[f"{level}_obj"][b, 0])
                m = obj.max().item()
                if m > max_conf:
                    max_conf = m
            if max_conf > conf_thresh:
                patch_tensor = batch[0][b].cpu()
                if len(candidates_heap) < top_k:
                    _heapq.heappush(candidates_heap,
                                    (max_conf, candidate_id, patch_tensor))
                elif max_conf > candidates_heap[0][0]:
                    _heapq.heapreplace(candidates_heap,
                                       (max_conf, candidate_id, patch_tensor))
                candidate_id += 1
            total_evaluated += 1

        if (batch_idx + 1) % 20 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    candidates_heap.sort(key=lambda x: x[0], reverse=True)
    n_fp = len(candidates_heap)
    fp_rate = n_fp / max(total_evaluated, 1)
    print(f"  Mined {n_fp} hard negatives from "
          f"{total_evaluated} crops (FP rate={fp_rate:.4f}), keeping top-{n_fp}")
    return [p for _, _, p in candidates_heap], n_fp


def save_epoch_checkpoint(ckpt_path, epoch, model, ema, optimizer, scheduler,
                           train_loss, val_f1, lr, grad_norm, output_stats,
                           diagnostics, step=0, best_val_f1=0.0):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None
                          else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_f1': val_f1,
        'lr': lr,
        'grad_norm': grad_norm,
        'output_stats': output_stats,
        'diagnostics': diagnostics,
        'step': step,
        'best_val_f1': best_val_f1,
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, ckpt_path)


def save_diagnostics_json(diag_path, epoch, diagnostics, output_stats,
                           val_results, train_loss, grad_norm, lr):
    with open(diag_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': float(train_loss),
            'grad_norm': float(grad_norm),
            'lr': float(lr),
            'val_results': val_results,
            'output_stats': output_stats,
            'diagnostics': diagnostics,
        }, f, indent=2, default=float)


def main():
    parser = argparse.ArgumentParser(description="FaceCNN v6.0 Training")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/face_cnn_v6_best.pth")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (auto-reduced if GPU < 5.5GB)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="pos_weight for P4 (must be >8 for bias to shift UP at init)")
    parser.add_argument("--pos-weight-p3", type=float, default=25.0,
                        help="pos_weight for P3 (medium faces)")
    parser.add_argument("--pos-weight-p2", type=float, default=50.0,
                        help="pos_weight for P2 (small faces)")
    parser.add_argument("--p3-obj-start", type=int, default=15,
                        help="Epoch at which P3 obj loss activates (0=immediate)")
    parser.add_argument("--p2-obj-start", type=int, default=30,
                        help="Epoch at which P2 obj loss activates (0=immediate)")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint file")
    parser.add_argument("--resume-lr-override", type=float, default=None,
                        help="Override LR on resume. If set, overwrites the "
                             "checkpoint's optimizer LR and creates a fresh "
                             "cosine scheduler from this value (e.g., 5e-4). "
                             "Essential for fine-tuning from a mid-training "
                             "checkpoint where the original LR is too high.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--validate-interval", type=int, default=1)
    parser.add_argument("--mine-interval", type=int, default=10)
    parser.add_argument("--mine-start", type=int, default=20)
    parser.add_argument("--mine-retrain", action="store_true",
                        help="Enable active fine-tuning on hard negatives")
    parser.add_argument("--ckpt-interval", type=int, default=5)
    parser.add_argument("--diag-interval", type=int, default=10)
    parser.add_argument("--flat-lr", action="store_true",
                        help="Keep LR constant (no decay) during training. "
                             "Recommended for fine-tuning from a converged "
                             "checkpoint with --resume-lr-override.")
    parser.add_argument("--mine-images", type=int, default=2000,
                        help="Number of images to mine hard negatives from")
    parser.add_argument("--mine-top-k", type=int, default=96,
                        help="Number of top hard negatives to keep per round")
    parser.add_argument("--hn-cache", type=str, default=None,
                        help="Path to persistent HN cache file. If set, HNs "
                             "accumulate across mining rounds (loaded at start, "
                             "saved after each round, kept sorted by confidence). "
                             "Retrain uses ALL accumulated HNs, not just latest.")
    parser.add_argument("--hn-cache-max", type=int, default=768,
                        help="Maximum accumulated HNs in cache (default 768)")
    parser.add_argument("--mine-mix-pos", type=int, default=0,
                        help="Number of positive batches to interleave during "
                             "HN retrain to prevent catastrophic forgetting. "
                             "0 disables. Recommended: 2-4.")
    parser.add_argument("--mine-val", action="store_true",
                        help="Mine hard negatives from validation set instead "
                             "of training set for added diversity")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({total_gb:.1f} GB)")
        if total_gb < 5.5 and args.batch_size > 8:
            old_bs = args.batch_size
            args.batch_size = 8
            print(f"  Auto batch-size: {old_bs} → 8 (insufficient VRAM for batch={old_bs})")

    from src.cv.face_detector_cnn import FaceFCNv5

    # ── Setup directories ──
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    diag_dir = os.path.join(out_dir, "v6_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "v6_epochs")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Header ──
    print("\n" + "=" * 70)
    print("  FaceCNN v6.0 — Full Retrain From Scratch")
    print("  Architecture: FaceFCNv5 (394K params, anchor-free FPN)")
    print("=" * 70)
    print(f"\n  Head init:     kaiming_normal + bias=-2.5 (v5: normal 0.01, -4.6)")
    print(f"  Loss:          BCEWithLogitsLoss(pos_weight=P4={args.pos_weight},P3={args.pos_weight_p3},P2={args.pos_weight_p2}) + GIoU")
    print(f"  Logit clamp:   [-10, 10] before BCE (prevents extreme gradients)")
    print(f"  NaN guard:     logit clamp + nan_to_num + grad NaN skip")
    print(f"  Optimizer:     diff-LR (bb={args.lr:.0e}, fpn={args.lr*2:.0e}, heads={args.lr*5:.0e})")
    print(f"  Head wd=0.0    FPN wd=1e-5    Backbone wd=1e-4")
    lr_schedule_desc = f"{args.warmup_epochs}-ep warmup → smooth cosine → 1e-5 (NO restart)"
    if args.flat_lr:
        lr_schedule_desc = "FLAT (constant LR, no decay — fine-tuning mode)"
    print(f"  LR schedule:   {lr_schedule_desc}")
    print(f"  Progressive:   P4=immediate, P3=ep{args.p3_obj_start}, P2=ep{args.p2_obj_start}")
    print(f"  EMA:           decay={args.ema_decay}  (BN buffer sync FIXED)")
    mining_desc = f"every {args.mine_interval} ep from ep {args.mine_start}"
    if args.mine_interval > 0:
        mining_desc += f"\n                 {args.mine_images} imgs, top-{args.mine_top_k} HNs"
        if args.hn_cache:
            mining_desc += f", cache={args.hn_cache} (max {args.hn_cache_max})"
        if args.mine_mix_pos > 0:
            mining_desc += f", mix-pos={args.mine_mix_pos}"
        if args.mine_val:
            mining_desc += ", val-set mining"
    mining_desc += "" if args.mine_retrain else " (diagnostic-only, --mine-retrain to enable active)"
    if args.mine_retrain:
        mining_desc = mining_desc.replace("(diagnostic-only, --mine-retrain to enable active)",
                                          "(active fine-tuning)")
    print(f"  Mining:        {mining_desc}")
    print(f"  Checkpoints:   every {args.ckpt_interval} ep + best")
    print(f"  Diagnostics:   every {args.diag_interval} ep (JSON + BN stats + output stats)")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Resume:        {args.resume if args.resume else 'no (training from scratch)'}")
    print(f"  Epochs:        {args.epochs}")

    # ── Data ──
    train_dataset = WiderFaceFPNDataset(
        args.data, "train", target_h=480, target_w=640, augment=True)
    val_dataset = WiderFaceFPNDataset(
        args.data, "val", target_h=480, target_w=640, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True,
                               drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    n_batches = len(train_loader)
    n_val_batches = len(val_loader)
    print(f"\n  Train: {len(train_dataset)} img | Batches/ep: {n_batches}")
    print(f"  Val:   {len(val_dataset)} img | Batches: {n_val_batches}")
    print(f"  Est. time: ~{args.epochs * (240/60):.0f} min (~{(args.epochs * 240/3600):.1f} h)")

    # ── Model ──
    model = FaceFCNv5().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {total_p:,} ({total_p * 4 / 1e6:.1f} MB FP32)")

    # ── Loss ──
    criterion_cls = {
        'p2': nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight_p2], device=device),
            reduction='mean'),
        'p3': nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight_p3], device=device),
            reduction='mean'),
        'p4': nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight], device=device),
            reduction='mean'),
    }
    criterion_bbox = AnchorFreeGIoULoss()
    print(f"  Loss: BCEWithLogitsLoss per-level:")
    print(f"    P4: pos_weight={args.pos_weight:.0f}  (active immediately)")
    print(f"    P3: pos_weight={args.pos_weight_p3:.0f} (activates epoch {args.p3_obj_start})")
    print(f"    P2: pos_weight={args.pos_weight_p2:.0f} (activates epoch {args.p2_obj_start})")
    print(f"    Progressive: backbone learns from P4 first; P3 joins ep{args.p3_obj_start}, P2 joins ep{args.p2_obj_start}")
    print(f"    Bbox heads (P2/P3/P4) always train — only obj loss is gated.")

    # ── Optimizer & Scheduler ──
    optimizer = get_optimizer(model, base_lr=args.lr)
    warmup_steps = args.warmup_epochs * n_batches
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-5)

    # ── EMA ──
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ── AMP ──
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    print("=" * 70)

    # ── Metrics tracking ──
    metrics_path = args.output.replace(".pth", "_metrics.csv")
    csv_headers = [
        "epoch", "train_loss",
        "train_obj_p2", "train_obj_p3", "train_obj_p4",
        "train_bbox_p2", "train_bbox_p3", "train_bbox_p4",
        "val_obj_p2", "val_obj_p3", "val_obj_p4",
        "val_f1_p2", "val_f1_p3", "val_f1_p4",
        "out_max_p4", "out_std_p4", "cells_above_03_p4",
        "head_bias_p4", "grad_norm", "lr", "epoch_time_s", "gpu_mem_mb", "mined",
    ]
    metrics_rows = []
    best_val_f1 = 0.0
    step = 0
    start_epoch = 0

    # ── Resume from checkpoint ──
    if args.resume:
        ckpt_path = os.path.abspath(args.resume)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        if ema is not None and 'ema_state_dict' in ckpt:
            ema.ema_model.load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        train_loss = ckpt.get('train_loss', 0.0)
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        step = ckpt.get('step', 0)
        print(f"\n  ★ RESUMED from epoch {start_epoch} | best F1={best_val_f1:.4f} | step={step}")
        print(f"  Checkpoint: {ckpt_path}")

        # ── LR Override on Resume ──
        if args.resume_lr_override is not None:
            mults = [1.0, 2.0, 5.0]
            for i, group in enumerate(optimizer.param_groups):
                group['lr'] = args.resume_lr_override * mults[i]
            if args.flat_lr:
                scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda _: 1.0)
                current_lr_info = "×".join(
                    f"{args.resume_lr_override * m:.1e}" for m in mults)
                print(f"  ★ LR OVERRIDE: {current_lr_info} (FLAT, no decay — fine-tuning mode)")
            else:
                schedule_epochs = args.epochs - start_epoch
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, schedule_epochs), eta_min=1e-5)
                current_lr_info = "×".join(
                    f"{args.resume_lr_override * m:.1e}" for m in mults)
                print(f"  ★ LR OVERRIDE: {current_lr_info} → 1e-5 over {schedule_epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()

        current_lr = optimizer.param_groups[-1]["lr"]

        # ── Progressive level gating ──
        p3_obj_active = epoch >= args.p3_obj_start
        p2_obj_active = epoch >= args.p2_obj_start

        # ── Hard-negative mining ──
        mined = 0
        if (args.mine_interval > 0 and epoch > 0
                and epoch >= args.mine_start
                and epoch % args.mine_interval == 0):
            print(f"\n  ── Mining epoch {epoch+1} ──")
            fresh_negs, mined = hard_negative_mining(
                model, args.data, device, num_images=args.mine_images,
                top_k=args.mine_top_k, use_val=args.mine_val)

            # ── HN accumulation: load cache, merge fresh, save ──
            acc_negs = []
            hn_cache_path = args.hn_cache
            if hn_cache_path is None:
                hn_cache_path = args.output.replace('.pth', '_hn_cache.pt')
            if os.path.exists(hn_cache_path):
                try:
                    prev = torch.load(hn_cache_path, map_location='cpu',
                                      weights_only=True)
                    acc_negs = list(prev) if isinstance(prev, list) else []
                except Exception:
                    acc_negs = []
            acc_negs.extend(fresh_negs)
            if len(acc_negs) > args.hn_cache_max:
                acc_negs = acc_negs[-args.hn_cache_max:]
            torch.save(acc_negs, hn_cache_path)
            print(f"  HN cache: {len(acc_negs)} accumulated "
                  f"(+{mined} fresh, max={args.hn_cache_max})")

            if acc_negs and args.mine_retrain:
                all_patches = torch.stack(acc_negs)
                B_all = all_patches.size(0)
                ng_h, ng_w = 480, 640

                # Shuffle accumulated HNs and train in chunks
                perm = torch.randperm(B_all)
                model.train()
                hn_steps = 0
                for si in range(0, B_all, 8):
                    ei = min(si + 8, B_all)
                    idx = perm[si:ei]
                    ch = idx.size(0)
                    p_chunk = all_patches[idx].to(device)
                    neg_targets_chunk = {
                        'p2': torch.zeros(ch, 1, ng_h // 2, ng_w // 2, device=device),
                        'p3': torch.zeros(ch, 1, ng_h // 4, ng_w // 4, device=device),
                        'p4': torch.zeros(ch, 1, ng_h // 8, ng_w // 8, device=device),
                    }
                    if ch == 0:
                        continue
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda', enabled=scaler is not None):
                        out_n = model(p_chunk)
                        mine_loss = torch.tensor(0.0, device=device)
                        for lv in ['p2', 'p3', 'p4']:
                            ml = criterion_cls[lv](
                                out_n[f'{lv}_obj'],
                                neg_targets_chunk[lv])
                            if (lv == 'p3' and not p3_obj_active) or \
                               (lv == 'p2' and not p2_obj_active):
                                ml = ml * 0.0
                            mine_loss = mine_loss + ml
                    if torch.isfinite(mine_loss):
                        if scaler is not None:
                            scaler.scale(mine_loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=5.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            mine_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=5.0)
                            optimizer.step()
                    hn_steps += 1
                    step += 1

                # ── Positive mixing: reinforce face detection ──
                if args.mine_mix_pos > 0:
                    mix_src = train_loader
                    pos_steps = 0
                    for pos_batch in mix_src:
                        if pos_steps >= args.mine_mix_pos:
                            break
                        (p_patches, p_hm2, p_bb2, p_hm3, p_bb3,
                         p_hm4, p_bb4) = pos_batch
                        p_patches = p_patches.to(device)
                        pos_targets = {
                            "p2": (p_hm2.to(device), p_bb2.to(device)),
                            "p3": (p_hm3.to(device), p_bb3.to(device)),
                            "p4": (p_hm4.to(device), p_bb4.to(device)),
                        }
                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda', enabled=scaler is not None):
                            p_out = model(p_patches)
                            pos_loss = torch.tensor(0.0, device=device)
                            for lv in ["p2", "p3", "p4"]:
                                po = torch.clamp(p_out[f"{lv}_obj"], -10.0, 10.0)
                                pt_hm, pt_bb = pos_targets[lv]
                                pm = (pt_hm > 0.5).float()
                                pl = criterion_cls[lv](po, pt_hm)
                                if (lv == 'p3' and not p3_obj_active) or \
                                   (lv == 'p2' and not p2_obj_active):
                                    pl = pl * 0.0
                                pos_loss = pos_loss + pl
                                if pm.sum() > 0:
                                    pb = criterion_bbox(
                                        p_out[f"{lv}_bbox"], pt_bb, pm,
                                        {"p2": 2, "p3": 4, "p4": 8}[lv])
                                    pos_loss = pos_loss + torch.nan_to_num(
                                        pb, nan=0.0)
                        if torch.isfinite(pos_loss):
                            if scaler is not None:
                                scaler.scale(pos_loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), max_norm=5.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                pos_loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), max_norm=5.0)
                                optimizer.step()
                        pos_steps += 1
                        step += 1

                print(f"  Fine-tuned on {len(acc_negs)} HNs ({hn_steps} steps)"
                      + (f" + {pos_steps} pos batches" if args.mine_mix_pos > 0
                         else ""))

        # ── Train ──
        train_loss, train_obj, train_bbox, grad_norm, step = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_bbox,
            device, scaler, warmup_steps=warmup_steps, base_lr=args.lr,
            step_start=step, p3_obj_active=p3_obj_active,
            p2_obj_active=p2_obj_active)

        if ema is not None:
            ema.update(model)

        epoch_time = time.time() - epoch_t0
        gpu_mem = 0
        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats()

        # ── Validate ──
        do_val = (epoch % args.validate_interval == 0) or (
            epoch == args.epochs - 1)
        val_results = {}
        output_stats = {}
        diagnostics = {}

        if do_val:
            val_results = validate(model, val_loader, criterion_cls, device)
            avg_f1 = float(np.mean([v["f1"] for v in val_results.values()]))

            # ── Diagnostics (periodic) ──
            if epoch == 0 or (epoch + 1) % args.diag_interval == 0 or \
               epoch == args.epochs - 1:
                print(f"  ── Collecting diagnostics (epoch {epoch+1}) ──")
                output_stats = compute_output_stats(
                    model if ema is None else ema.ema_model,
                    val_loader, device, num_batches=20)
                diagnostics = collect_diagnostics(
                    model if ema is None else ema.ema_model)

                diag_path = os.path.join(
                    diag_dir, f"v6_epoch_{epoch+1:03d}.json")
                save_diagnostics_json(
                    diag_path, epoch + 1, diagnostics, output_stats,
                    {k: {"f1": v["f1"], "obj_loss": v["obj_loss"]}
                     for k, v in val_results.items()},
                    train_loss, grad_norm, current_lr)
                print(f"    Saved: {diag_path}")
                if output_stats:
                    p4_s = output_stats.get('p4', {})
                    print(f"    P4 max sigmoid: {p4_s.get('max_max', 0):.4f} | "
                          f"cells>0.03: {p4_s.get('cells_above_03_per_image', 0):.1f} | "
                          f"cells>0.05: {p4_s.get('cells_above_05_per_image', 0):.1f}")

            # ── Best checkpoint ──
            if avg_f1 > best_val_f1 + 1e-8:
                best_val_f1 = avg_f1
                save_epoch_checkpoint(
                    args.output, epoch + 1, model, ema, optimizer, scheduler,
                    train_loss, avg_f1, current_lr, grad_norm,
                    output_stats, diagnostics, step=step,
                    best_val_f1=best_val_f1)
                print(f"  ★ BEST checkpoint (mean F1={avg_f1:.4f}) at epoch {epoch+1}")

        # ── Epoch checkpoint ──
        if (epoch + 1) % args.ckpt_interval == 0 or epoch == args.epochs - 1:
            epoch_ckpt = os.path.join(
                ckpt_dir, f"v6_epoch_{epoch+1:03d}.pth")
            save_epoch_checkpoint(
                epoch_ckpt, epoch + 1, model, ema, optimizer, scheduler,
                train_loss, avg_f1 if do_val else 0.0,
                current_lr, grad_norm, output_stats, diagnostics,
                step=step, best_val_f1=best_val_f1)

        # ── CSV row ──
        p4_out_max = output_stats.get('p4', {}).get('max_max', 0.0)
        p4_out_std = output_stats.get('p4', {}).get('std_mean', 0.0)
        p4_cells_03 = output_stats.get('p4', {}).get(
            'cells_above_03_per_image', 0.0)
        try:
            p4_bias = float(model.get_parameter('head_p4.obj_pred.bias').item())
        except (AttributeError, RuntimeError):
            p4_bias = diagnostics.get('head_biases', {}).get(
                'head_p4.obj_pred', {}).get('value', float('nan'))

        if do_val:
            row = [
                epoch + 1, f"{train_loss:.4f}",
                f"{train_obj['p2']:.4f}", f"{train_obj['p3']:.4f}", f"{train_obj['p4']:.4f}",
                f"{train_bbox['p2']:.4f}", f"{train_bbox['p3']:.4f}", f"{train_bbox['p4']:.4f}",
                f"{val_results['p2']['obj_loss']:.4f}", f"{val_results['p3']['obj_loss']:.4f}", f"{val_results['p4']['obj_loss']:.4f}",
                f"{val_results['p2']['f1']:.4f}", f"{val_results['p3']['f1']:.4f}", f"{val_results['p4']['f1']:.4f}",
                f"{p4_out_max:.4f}", f"{p4_out_std:.4f}", f"{p4_cells_03:.1f}",
                f"{p4_bias:.4f}", f"{grad_norm:.4f}", f"{current_lr:.8f}",
                f"{epoch_time:.2f}", f"{gpu_mem:.1f}", str(mined),
            ]
        else:
            row = [epoch + 1, f"{train_loss:.4f}"] + ["skip"] * 21

        # ── Print ──
        f1_str = " | ".join(
            f"{lv.upper()}={val_results[lv]['f1']:.3f}"
            for lv in ['p2', 'p3', 'p4']) if do_val else "N/A"
        diag_str = ""
        if output_stats:
            diag_str = (f" | P4 max={p4_out_max:.3f} "
                        f"cells>0.3={p4_cells_03:.0f}")
        star = " ★" if do_val and abs(avg_f1 - best_val_f1) < 1e-8 else ""
        mine_str = f" | mined={mined}" if mined > 0 else ""
        mine_str += " (retrain)" if mined > 0 and args.mine_retrain else (" (diag)" if mined > 0 else "")

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Grad: {grad_norm:.2f} | "
            f"F1: {f1_str}{diag_str} | "
            f"LR={current_lr:.1e} | t={epoch_time:.0f}s{star}{mine_str}"
        )

        metrics_rows.append(row)
        with open(metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_headers)
            w.writerows(metrics_rows)

        if epoch >= args.warmup_epochs and not args.flat_lr:
            scheduler.step()

    # ── Final summary ──
    print(f"\n{'=' * 70}")
    print(f"  Training complete.")
    print(f"  Best mean heatmap F1: {best_val_f1:.4f}")
    print(f"  Checkpoint:           {args.output}")
    print(f"  Epoch checkpoints:    {ckpt_dir}/")
    print(f"  Diagnostics:          {diag_dir}/")
    print(f"  Metrics CSV:          {metrics_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
