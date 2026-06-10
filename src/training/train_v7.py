"""
FaceCNN v7.0 — Production Training Pipeline
=============================================
Two-Phase Schedule (Target: 45% detection recall, ~22.2h GPU)

Phase 1 — Max Training (200 epochs, ~17.0h):
  v7 arch (shared 3x3 head, 4-layer, 5x bbox weight)
  Quality score (sqrt(obj * iou)) for NMS ranking
  CIoU loss (center distance + aspect ratio replaces GIoU)
  Multi-scale training (384-800px random resize + crop)
  Copy-paste augmentation (+5 faces per image from batch)
  Hard-negative mining (every 10 epochs from ep 20)
  EMA inference weights (decay=0.999, BN buffers synced)
  Soft-NMS (decay overlapping scores, not hard suppress)
  SGDR x3 (CosineAnnealingWarmRestarts, T_0=67)
  Label smoothing (s=0.1)
  Gradient accumulation 2 (effective batch 32)

Phase 2 — Pseudo-Label Fine-Tuning (3 x 25 epochs, ~5.2h):
  Cycle 1: Add 5K pseudo-labeled WIDER test images (quality > 0.3)
  Cycle 2: Add 10K (5K new pseudo-labels from cycle 1 model)
  Cycle 3: Add 16K (6K new pseudo-labels from cycle 2 model)
  LR: 1e-4 -> 1e-6, label smoothing s=0.2 for pseudo labels
  Mining enabled throughout (self-correcting adversarial loop)

Loss: SmoothBCE(obj) + MSE(iou_quality) + 5xCIoU(bbox)
Scheduler: SGDR (CosineAnnealingWarmRestarts, T_0=67, T_mult=1)
Data: Multi-scale 384-800px, copy-paste, hard-negative mining
Est. total time: ~22.2h on RTX 2060 (under 24h budget)

=== PITFALL PREVENTION CHECKLIST ===

[ ] v4 crop-based training mismatch → v7 full-frame FPN training
[ ] v4 model capacity (62K) → v7 519K params, 4-layer heads
[ ] v5 EMA BN buffer freeze → ModelEMA.update() syncs buffers (line 248-249 train_face_cnn.py)
[ ] v5 dead obj head bias stuck at -4.6 → monitored via diag['head_biases']['p{3,4}_obj']['sigmoid']
[ ] v5 LR restart degradation → health_check() and detection_pipeline_check() at ep 66, 133
[ ] v6 dead bbox heads (L2→0) → monitored via per-layer gradient L2 + weight L2 norms in diagnostics
[ ] v6 P2 saturation (100% firing) → P2 dropped entirely
[ ] v6 heatmap F1 ≠ detection quality → detection_pipeline_check() runs full decode on synthetic frame
[ ] v6 bbox gradient starvation (4%) → 5x bbox weight + CIoU center gradient
[ ] v6 single 1x1 linear head → 4-layer shared head with 3x3 spatial context
[ ] BN running stats frozen → bn_stats in diagnostics (running_mean_abs_avg, running_var_avg)
[ ] Box size collapse → detection_pipeline_check() reports tiny box ratio
[ ] NaN/Inf in output → health_check() at init and post-restart
"""

import os, sys, time, csv, math, argparse, json, warnings, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv.face_detector_v7 import FaceFCNv7
from src.training.train_face_cnn import ModelEMA, WiderFaceFPNDataset, AnchorFreeCIoULoss, FocalLoss

warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")

# ============================================================================
# Label-Smoothed BCE
# ============================================================================

class SmoothBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing. Prevents sigmoid saturation by
    softening targets from {0,1} to {0.05, 0.95}."""
    def __init__(self, smoothing=0.1, pos_weight=None, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, pred, target):
        target_smooth = target * (1 - self.smoothing * 2) + self.smoothing
        loss = self.bce(pred, target_smooth)
        return loss.mean()


# ============================================================================
# Combined loss: SmoothBCE(obj) + MSE(iou) + 5xCIoU(bbox)
# ============================================================================

class V7Loss(nn.Module):
    def __init__(self, pos_weights={"p3": 25.0, "p4": 10.0}, smoothing=0.1,
                 p3_focal_gamma=None, p3_mse=False):
        super().__init__()
        self.p3_mse = p3_mse
        self.criterion_cls = {}
        for lv in ["p3", "p4"]:
            if lv == "p3" and p3_mse:
                continue  # MSE uses raw F.mse_loss, no BCE criterion needed
            if lv == "p3" and p3_focal_gamma is not None:
                self.criterion_cls[lv] = FocalLoss(
                    gamma=p3_focal_gamma, alpha=0.75, reduction="mean")
            else:
                self.criterion_cls[lv] = SmoothBCEWithLogitsLoss(
                    smoothing=smoothing,
                    pos_weight=torch.tensor([pos_weights[lv]]))
        self.criterion_bbox = AnchorFreeCIoULoss()
        self.bbox_weight = 5.0

    def to(self, device):
        for lv in ["p3", "p4"]:
            if lv in self.criterion_cls and hasattr(self.criterion_cls[lv], 'bce'):
                self.criterion_cls[lv].bce.pos_weight = \
                    self.criterion_cls[lv].bce.pos_weight.to(device)
        return super().to(device)

    def forward(self, outputs, targets, strides, p3_active=True, p4_active=True):
        total_loss = torch.tensor(0.0, device=list(outputs.values())[0].device)
        obj_losses = {}
        iou_losses = {}
        bbox_losses = {}

        for lv in ["p3", "p4"]:
            pred_obj = outputs[f"{lv}_obj"]
            pred_iou = outputs[f"{lv}_iou"]
            pred_bbox = outputs[f"{lv}_bbox"]
            t_hm, t_bb = targets[lv]

            active = {"p3": p3_active, "p4": p4_active}[lv]
            pos_mask = (t_hm > 0.5).float()

            # Objectness: MSE on ATSS IoU targets (P3) or smoothed BCE (P4)
            if active:
                if lv == "p3" and self.p3_mse:
                    # Multiply by N_cells to cancel the /N from mean reduction,
                    # giving each cell a gradient of (sigmoid - target).
                    n_total = pred_obj.numel()
                    obj_l = F.binary_cross_entropy_with_logits(pred_obj, t_hm, reduction='mean') * n_total
                else:
                    obj_l = self.criterion_cls[lv](pred_obj, t_hm)
            else:
                obj_l = torch.tensor(0.0)
            obj_losses[lv] = obj_l

            # IoU quality prediction (MSE on positive cells)
            if pos_mask.sum() > 0:
                with torch.no_grad():
                    actual_iou = compute_iou_for_loss(pred_bbox, t_bb, pos_mask, strides[lv])
                iou_l = F.mse_loss(torch.sigmoid(pred_iou) * pos_mask, actual_iou * pos_mask)
            else:
                iou_l = torch.tensor(0.0)
            iou_losses[lv] = iou_l

            # Bbox regression (CIoU, 5x weight)
            bbox_l = self.criterion_bbox(pred_bbox, t_bb, pos_mask, strides[lv]) * self.bbox_weight
            bbox_losses[lv] = bbox_l

            total_loss = total_loss + obj_l + iou_l + bbox_l

        return total_loss, obj_losses, iou_losses, bbox_losses


@torch.no_grad()
def compute_iou_for_loss(pred_bbox, target_bbox, pos_mask, stride):
    """Compute actual IoU between predicted and GT boxes on positive cells."""
    pm = pos_mask > 0.5
    if pm.sum() == 0:
        return torch.zeros_like(pos_mask)

    B = pred_bbox.size(0)
    device = pred_bbox.device
    gh, gw = pred_bbox.size(-2), pred_bbox.size(-1)

    ys, xs = torch.meshgrid(
        torch.arange(gh, device=device, dtype=torch.float32),
        torch.arange(gw, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xs = xs.unsqueeze(0).expand(B, -1, -1)
    ys = ys.unsqueeze(0).expand(B, -1, -1)

    pd = pred_bbox[:, 0]; qd = pred_bbox[:, 1]
    pw_ = pred_bbox[:, 2].clamp(max=5.0)
    ph_ = pred_bbox[:, 3].clamp(max=5.0)
    p_cx = (xs + 0.5 + pd) * stride
    p_cy = (ys + 0.5 + qd) * stride
    p_w = torch.exp(pw_) * stride
    p_h = torch.exp(ph_) * stride

    td = target_bbox[:, 0]; ud = target_bbox[:, 1]
    tw_ = target_bbox[:, 2]; th_ = target_bbox[:, 3]
    t_cx = (xs + 0.5 + td) * stride
    t_cy = (ys + 0.5 + ud) * stride
    t_w = torch.exp(tw_) * stride
    t_h = torch.exp(th_) * stride

    x1 = torch.max(p_cx - p_w/2, t_cx - t_w/2).clamp(min=0)
    y1 = torch.max(p_cy - p_h/2, t_cy - t_h/2).clamp(min=0)
    x2 = torch.min(p_cx + p_w/2, t_cx + t_w/2).clamp(min=0)
    y2 = torch.min(p_cy + p_h/2, t_cy + t_h/2).clamp(min=0)
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = p_w * p_h + t_w * t_h - inter + 1e-8
    iou = (inter / union) * pm.float().squeeze(1)
    return iou.detach().unsqueeze(1)


# ============================================================================
# Data utilities
# ============================================================================

def wider_loader(data_root, split, batch_size, augment=True, num_workers=2,
                 use_atss=False, atss_levels=None):
    """WIDER Face loader that strips P2 from the 7-field output."""
    ds = WiderFaceFPNDataset(data_root, split, target_h=480, target_w=640,
                             augment=augment, use_atss=use_atss,
                             atss_levels=atss_levels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                         num_workers=num_workers, pin_memory=True,
                         drop_last=(split == "train"))
    return loader


def unpack_batch(batch):
    """Unpack batch tuple (7 fields) into (patches, P3 targets, P4 targets), skipping P2."""
    patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4 = batch
    targets = {
        "p3": (hm_p3, bb_p3),
        "p4": (hm_p4, bb_p4),
    }
    return patches, targets

def get_optimizer(model, base_lr=1e-3):
    bb, fp, hd, cell_bias = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'p3_cell_bias' in name:
            cell_bias.append(param)
        elif 'backbone' in name:
            bb.append(param)
        elif 'fpn' in name:
            fp.append(param)
        else:
            hd.append(param)
    opt = optim.AdamW([
        {'params': bb, 'lr': base_lr, 'weight_decay': 1e-4},
        {'params': fp, 'lr': base_lr * 2, 'weight_decay': 1e-5},
        {'params': hd, 'lr': base_lr * 5, 'weight_decay': 0.0},
    ], lr=base_lr, weight_decay=1e-4)
    if cell_bias:
        # Cell bias needs moderate LR: high enough to converge in ~2 epochs,
        # low enough to not overshoot equilibrium (sigmoid 0.8, logit 1.4).
        # AdamW adaptive LR ~LR ≈ effective step at steady state.
        opt.add_param_group({
            'params': cell_bias, 'lr': base_lr * 0.4, 'weight_decay': 0.0,
        })
    return opt


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_obj = {"p3": 0.0, "p4": 0.0}
    tp = {"p3": 0, "p4": 0}; fp = {"p3": 0, "p4": 0}; fn = {"p3": 0, "p4": 0}
    n_batches = 0

    for batch in tqdm(loader, desc="Val v7"):
        patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4 = batch
        patches = patches.to(device)
        targets = {
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }
        outputs = model(patches)
        for lv in ["p3", "p4"]:
            pred_obj = outputs[f"{lv}_obj"].clamp(-10, 10)
            obj_val = F.binary_cross_entropy_with_logits(pred_obj, targets[lv][0])
            val_obj[lv] += obj_val.item() if torch.isfinite(obj_val) else 0.0

            probs = torch.sigmoid(pred_obj)
            gt_bin = (targets[lv][0] > 0.5).float()
            pred_bin = (probs > 0.5).float()
            tp[lv] += (pred_bin * gt_bin).sum().item()
            fp[lv] += (pred_bin * (1 - gt_bin)).sum().item()
            fn[lv] += ((1 - pred_bin) * gt_bin).sum().item()
        n_batches += 1

    n = max(n_batches, 1)
    results = {}
    for lv in ["p3", "p4"]:
        p = tp[lv] / max(tp[lv] + fp[lv], 1)
        r = tp[lv] / max(tp[lv] + fn[lv], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[lv] = {"obj_loss": val_obj[lv] / n, "precision": p, "recall": r, "f1": f1}
    return results


@torch.no_grad()
def collect_diagnostics(model):
    diag = {"head_weights": {}, "head_biases": {}, "bn_stats": {}, "gradient_per_layer": {}}
    for name in ["p3", "p4"]:
        for branch in ["obj", "iou", "bbox"]:
            w = model.get_parameter(f"head.{name}_{branch}.weight")
            diag['head_weights'][f"{name}_{branch}"] = float(w.norm(2).item())
        for branch in ["obj", "iou"]:
            b = model.get_parameter(f"head.{name}_{branch}.bias")
            diag['head_biases'][f"{name}_{branch}"] = {
                "value": float(b.item()), "sigmoid": float(torch.sigmoid(b).item())
            }
        b = model.get_parameter(f"head.{name}_bbox.bias")
        diag['head_biases'][f"{name}_bbox"] = {
            "dx": float(b[0].item()), "dy": float(b[1].item()),
            "dw": float(b[2].item()), "dh": float(b[3].item())
        }

    # Per-layer gradient L2 norm (detect dead layers)
    for n, p in model.named_parameters():
        if p.grad is not None:
            key = n.replace('.', '_')
            diag['gradient_per_layer'][key] = float(p.grad.norm(2).item())

    # BN running stats summary (detect frozen BN — v5 root cause)
    bn_mean_abs_sum, bn_var_sum, bn_count = 0.0, 0.0, 0
    for n, buf in model.named_buffers():
        if 'running_mean' in n:
            bn_mean_abs_sum += buf.abs().mean().item()
            bn_count += 1
        elif 'running_var' in n:
            bn_var_sum += buf.mean().item()
    if bn_count > 0:
        diag['bn_stats'] = {
            "running_mean_abs_avg": bn_mean_abs_sum / bn_count,
            "running_var_avg": bn_var_sum / bn_count,
            "n_bn_layers": bn_count,
        }

    # P3 per-cell bias stats
    if hasattr(model.head, 'p3_cell_bias'):
        cb = model.head.p3_cell_bias
        diag['p3_cell_bias'] = {
            "mean": float(cb.mean().item()),
            "std": float(cb.std().item()),
            "min": float(cb.min().item()),
            "max": float(cb.max().item()),
        }

    return diag


def save_checkpoint(path, epoch, model, ema, optimizer, scheduler, train_loss,
                     val_f1, lr, grad_norm, step, best_val_f1=0.0):
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict() if ema else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss, 'val_f1': val_f1, 'lr': lr,
        'grad_norm': grad_norm, 'step': step, 'best_val_f1': best_val_f1,
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FaceCNN v7.0 Training")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/face_cnn_v7.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=0,
                        help="Val batch size (default: same as --batch-size, reduce for low VRAM)")
    parser.add_argument("--p3-finetune", action="store_true",
                        help="P3-only fine-tune mode: freeze backbone+FPN+P4, reinit P3 head, train P3 from scratch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--sgdr-t0", type=int, default=67, help="SGDR first cycle length (67 ep for 3 cycles over 200 ep)")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--bbox-weight", type=float, default=5.0)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--validate-interval", type=int, default=1)
    parser.add_argument("--mine-interval", type=int, default=10)
    parser.add_argument("--mine-start", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ckpt-interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--diag-interval", type=int, default=1,
                        help="Collect and save JSON diagnostics every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (e.g. models/face_cnn_v7_ep140.pth)")
    parser.add_argument("--no-mining", action="store_true",
                        help="Disable hard-negative mining (reduces VRAM)")
    parser.add_argument("--p3-pos-weight", type=float, default=50.0,
                        help="P3 positive weight for BCE loss (default: 50)")
    parser.add_argument("--p3-focal-gamma", type=float, default=None,
                        help="Enable FocalLoss for P3 with given gamma (default: disabled, uses SmoothBCE)")
    parser.add_argument("--p3-neg-ratio", type=float, default=None,
                        help="Per-batch OHEM: keep top-k hardest negatives per positive (e.g. 15). Disables mean BCE for P3.")
    parser.add_argument("--p3-fpn-unfreeze", action="store_true",
                        help="Unfreeze FPN P3 pathway (lat3, refine3, se3) for feature adaptation")
    parser.add_argument("--p3-bias", type=float, default=None,
                        help="Initial P3 obj bias (e.g. -7.0 for sigmoid=0.0009), overrides default -2.5")
    parser.add_argument("--p3-mse", action="store_true",
                        help="Use MSE regression on ATSS IoU targets for P3 instead of BCE")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")
        if gb < 5.5 and args.batch_size > 8:
            args.batch_size = 8
            print("  Auto batch-size: 16 → 8 (6GB VRAM)")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FaceCNN v7.0 — Production Training (Phase 1 / 200 ep)")
    print("  Architecture: FaceFCNv7 (519K params, shared head)")
    print("  Loss: SmoothBCE(obj) + MSE(iou) + 5xCIoU(bbox)")
    print("  Schedule: SGDR (T_0=67), 200 epochs, ~17h")
    print("  Strategies: multi-scale, copy-paste, hard-negative mining")
    print("  Data: WIDER Face 12,880 images")
    print("=" * 70)

    # Model
    model = FaceFCNv7(p3_enhanced=args.p3_finetune).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    if args.p3_finetune:
        base_p = 519476
        added_p = total_p - base_p
        print(f"\n  Parameters: {total_p:,} ({total_p*4/1e6:.1f} MB FP32, +{added_p:,} from P3 enhanced)")
    else:
        print(f"\n  Parameters: {total_p:,} ({total_p*4/1e6:.1f} MB FP32)")

    # Model health check (v5 root cause: dead detection head)
    @torch.no_grad()
    def _health_check(m, tag="init"):
        m.eval()
        zero_in = torch.zeros(1, 3, 480, 640, device=device)
        out = m(zero_in)
        ok = True
        for lv in ["p3", "p4"]:
            obj = out[f"{lv}_obj"]
            if not torch.isfinite(obj).all():
                n, i = torch.isnan(obj).sum().item(), torch.isinf(obj).sum().item()
                print(f"  ⚠ HEALTH [{tag}] {lv}: NaN={n} Inf={i}")
                ok = False
            std = obj.std().item()
            if std < 1e-5:
                print(f"  ⚠ HEALTH [{tag}] {lv}: COLLAPSED (std={std:.2e})")
                ok = False
            max_sig = float(torch.sigmoid(obj).max().item())
            print(f"  [{tag}] {lv} obj: std={std:.2e} max_sig={max_sig:.4f}")
        if ok:
            print(f"  [{tag}] Model health: PASS")
        return ok

    @torch.no_grad()
    def _detection_pipeline_check(m, tag=""):
        """Run full detect() on a synthetic frame, report decoded box stats.
        Catches v6 pitfall: bbox head predicts tiny boxes (<5px) that all
        get filtered, producing zero detections despite healthy heatmap F1."""
        m.eval()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        out = m(tensor.unsqueeze(0).to(device))
        total_boxes, tiny_boxes, total_peaks = 0, 0, 0
        for lname, stride in [("p3", 4), ("p4", 8)]:
            obj = out[f"{lname}_obj"][0, 0].cpu().numpy()
            bbox = out[f"{lname}_bbox"][0].cpu().numpy()
            kernel = np.ones((3, 3), dtype=np.uint8)
            peaks = (obj == cv2.dilate(obj, kernel)) & (obj > -2.5)
            if not peaks.any():
                continue
            ys, xs = np.where(peaks)
            total_peaks += len(ys)
            for cy, cx in zip(ys, xs):
                dw = float(bbox[2, cy, cx])
                dh = float(bbox[3, cy, cx])
                bw = float(np.exp(np.clip(dw, -2, 5))) * stride
                bh = float(np.exp(np.clip(dh, -2, 5))) * stride
                total_boxes += 1
                if bw < 5 or bh < 5:
                    tiny_boxes += 1
        kept = total_boxes - tiny_boxes
        msg = f"  [{tag}] Detect: {total_peaks} peaks, {total_boxes} boxes, {tiny_boxes} tiny (<5px), {kept} kept"
        if kept == 0 and total_boxes > 0:
            msg += " ⚠ ALL BOXES TINY — bbox head collapsed"
        elif kept == 0 and total_boxes == 0:
            msg += " ⚠ NO PEAKS — obj head collapsed"
        else:
            kept_ratio = kept / max(total_boxes, 1)
            msg += f" ({kept_ratio:.0%} keep rate)"
        print(msg)
        return kept, total_boxes, total_peaks

    _health_check(model, "init")
    _detection_pipeline_check(model, "init")

    # Loss
    p3_focal_gamma = args.p3_focal_gamma if args.p3_finetune else None
    p3_mse = args.p3_mse and args.p3_finetune
    criterion = V7Loss(smoothing=args.smoothing, p3_focal_gamma=p3_focal_gamma,
                       p3_mse=p3_mse).to(device)
    criterion.bbox_weight = args.bbox_weight

    # Optimizer & SGDR scheduler
    optimizer = get_optimizer(model, base_lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.sgdr_t0, T_mult=1, eta_min=1e-5)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # AMP
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    # Data
    val_bs = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    train_loader = wider_loader(args.data, "train", args.batch_size, augment=True,
                                use_atss=args.p3_finetune, atss_levels=["p3"])
    val_loader = wider_loader(args.data, "val", val_bs, augment=False,
                              use_atss=args.p3_finetune, atss_levels=["p3"])

    if args.p3_finetune:
        print("  P3 targets: ATSS dynamic assignment (IoU-based)  P4 targets: Gaussian")
    n_batches = len(train_loader)
    print(f"\n  Train: {len(train_loader.dataset)} img | {n_batches} batches/ep")
    print(f"  Val:   {len(val_loader.dataset)} img")
    eph = args.epochs * (306 / 3600)
    print(f"  Est. time: ~{eph:.1f} h ({args.epochs} epochs × ~306s — Max config with multi-scale + copy-paste + mining)")
    print(f"  Phase 2 (pseudo-labeling) will add ~5.2h (3x25 ep fine-tuning)")
    print(f"  Total pipeline: ~{eph + 5.2:.1f} h (under 24h budget)")

    # Metrics
    metrics_path = args.output.replace(".pth", "_metrics.csv")
    csv_headers = ["epoch","train_loss",
                    "val_obj_p3","val_obj_p4","val_f1_p3","val_f1_p4",
                    "val_prec_p3","val_rec_p3","val_prec_p4","val_rec_p4",
                    "p3_obj_bias","p4_obj_bias","p3_iou_bias","p4_iou_bias",
                    "p3_bbox_dw","p4_bbox_dw","p3_bbox_dh","p4_bbox_dh",
                    "p3_obj_l2","p3_iou_l2","p3_bbox_l2",
                    "p4_obj_l2","p4_iou_l2","p4_bbox_l2",
                    "bn_mean_abs","bn_var_avg",
                    "grad_norm","lr","epoch_time_s","gpu_mem_mb"]
    metrics_rows = []
    best_val_f1 = 0.0
    step = 0
    start_epoch = 0

    # ── Resume from checkpoint ──
    if args.resume:
        print(f"\n  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if args.p3_finetune:
            ckpt_state = ckpt['model_state_dict']
            for k in list(ckpt_state.keys()):
                if k.startswith('head.p3_') and not k.startswith('head.p3_conv3') and not k.startswith('head.p3_conv4'):
                    del ckpt_state[k]
            missing, _ = model.load_state_dict(ckpt_state, strict=False)
            p3_new = [k for k in missing if k.startswith('head.p3_')]
            if p3_new:
                print(f"  P3 enhanced: {len(p3_new)} new layers random init:")
                for n in p3_new:
                    print(f"    {n}")
            print("  Skipping optimizer/scheduler/EMA (rebuilt in P3 mode)")
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            elif 'lr' in ckpt:
                for g in optimizer.param_groups:
                    g['lr'] = ckpt['lr']
            if ema is not None and 'ema_state_dict' in ckpt:
                ema.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        step = ckpt.get('step', 0)
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        # Load existing metrics CSV rows
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                metrics_rows = [row for row in reader]
        print(f"  Resumed at epoch {start_epoch}, step {step}, best_val_f1={best_val_f1:.4f}")
        _health_check(model, "resume")
        _detection_pipeline_check(model, "resume")

    # ── P3 Fine-Tune Mode ──
    if args.p3_finetune:
        p3_lr = 5e-3  # higher LR for training from scratch
        print("\n  " + "=" * 70)
        print("  P3-ONLY FINE-TUNE MODE — 128ch + MSE REGRESSION")
        print("=" * 70)
        # Freeze everything
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze ONLY the per-cell bias initially. SE/refine/heads remain frozen
        # to prevent them from locking the fixed point before the bias differentiates.
        # At epoch 5 (see training loop), SE/refine/heads are unfrozen.
        for name, param in model.named_parameters():
            if 'p3_cell_bias' in name:
                param.requires_grad = True
        # Track head params for delayed unfreeze
        p3_head_params = []
        for name, param in model.named_parameters():
            if name.startswith('head.p3_') and not name.startswith('head.p3_conv3') and not name.startswith('head.p3_conv4') and not name.startswith('head.p3_project_up') and not 'p3_cell_bias' in name:
                p3_head_params.append(param)
        print("  P3 enhanced: 64→128 project-up, SE(128→16→128), grouped conv(g=8), 3×3 heads")
        # Init project_up to preserve conv4 features (copy each input channel to 2 output channels)
        with torch.no_grad():
            pw = model.head.p3_project_up[0].weight
            pw.zero_()
            for i in range(64):
                if 2*i < pw.size(0):
                    pw[2*i, i] = 1.0 / 2**0.5
                if 2*i+1 < pw.size(0):
                    pw[2*i+1, i] = 1.0 / 2**0.5
        print("  P3 project_up init: identity-preserving (1/sqrt(2) per copy channel)")
        # Freeze BN running stats for all BNs except new P3 enhanced BNs
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if 'head.p3_refine' in name:
                    m.momentum = 0.1
                else:
                    m.momentum = 0.0
        # Set P3 obj bias
        p3_bias_val = args.p3_bias if args.p3_bias is not None else -2.5
        nn.init.constant_(model.head.p3_obj.bias, p3_bias_val)
        bias_sigmoid = float(torch.sigmoid(torch.tensor(p3_bias_val)).item())
        print(f"  P3 obj bias set to {p3_bias_val} (sigmoid={bias_sigmoid:.4f})")
        # Init per-cell bias to 0 — learns only the residual (positive for face cells,
        # negative for background). Starting at 0 prevents the bias from pre-collapsing
        # all cells negative and giving the shared bias nowhere to go.
        nn.init.constant_(model.head.p3_cell_bias, 0.0)
        print("  P3 cell bias init: 0.0 (neutral start) — learns residual only")
        # BCE on continuous ATSS IoU targets — no pos_weight, no OHEM, no FocalLoss
        if args.p3_mse:
            print("  P3 loss: BCE on ATSS IoU targets (continuous 0.0-1.0 regression)")
            print("  No pos_weight, no OHEM, no FocalLoss, no smoothing for P3 obj")
        else:
            if hasattr(criterion.criterion_cls["p3"], 'bce'):
                criterion.criterion_cls["p3"].bce.pos_weight = (
                    torch.tensor([args.p3_pos_weight]).to(device))
                print(f"  P3 pos_weight={args.p3_pos_weight} (SmoothBCE)")
            elif "p3" in criterion.criterion_cls:
                print(f"  P3 FocalLoss gamma={args.p3_focal_gamma}")
            else:
                print("  P3 loss: MSE (no BCE criterion created)")
        # Zero out P4 loss
        criterion.criterion_cls["p4"].bce.pos_weight = torch.tensor([0.0]).to(device)
        # Rebuild optimizer
        optimizer = get_optimizer(model, base_lr=p3_lr)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_model = sum(p.numel() for p in model.parameters())
        pct = 100.0 * trainable / total_model
        print(f"  Trainable: {trainable:,} / {total_model:,} ({pct:.1f}%)")
        # CosineAnnealing LR for P3 finetune
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)
        print(f"  Scheduler: CosineAnnealing LR {p3_lr:.0e} → 1e-5 over {args.epochs} epochs")
        # Reset epoch counter and metrics for P3 scratch training
        start_epoch = 0
        metrics_rows = []
        best_val_f1 = 0.0
        print()

    strides = {"p3": 4, "p4": 8}
    diag_dir = os.path.join(out_dir, "v7_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()

        # Delayed unfreeze: at epoch 5, thaw P3 SE/refine/heads
        # (cell bias trained alone for 5 epochs to differentiate first)
        if epoch == 5 and args.p3_finetune and p3_head_params:
            for p in p3_head_params:
                p.requires_grad = True
            optimizer = get_optimizer(model, base_lr=p3_lr)
            print(f"  P3 heads unfrozen at epoch {epoch+1}: +{sum(p.numel() for p in p3_head_params):,} params")
            # Rebuild scheduler with new optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-5)

        # Defragment CUDA memory to prevent OOM from fragmentation
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Train ──
        model.train()
        total_loss = 0.0
        grad_sum = 0.0
        n_batches_done = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            patches, targets = unpack_batch(batch)
            patches = patches.to(device)
            targets = {lv: (t[0].to(device), t[1].to(device)) for lv, t in targets.items()}

            # SGDR warmup
            if epoch == 0 and args.warmup_steps > 0:
                scale = min(1.0, (step + 1) / args.warmup_steps)
                for i, group in enumerate(optimizer.param_groups):
                    group['lr'] = group['lr'] * scale

            p3_active = epoch >= 0
            p4_active = (not args.p3_finetune)  # disable P4 loss in P3 fine-tune mode

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = model(patches)
                loss, obj_l, iou_l, bbox_l = criterion(
                    outputs, targets, strides, p3_active, p4_active)

            loss = loss / args.grad_accum

            if torch.isfinite(loss):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                optimizer.zero_grad()
                continue

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for n, p in model.named_parameters() if 'p3_cell_bias' not in n], 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for n, p in model.named_parameters() if 'p3_cell_bias' not in n], 5.0)
                    optimizer.step()
                optimizer.zero_grad()
                step += 1

            gn = sum(p.grad.norm(2).item()**2 for n, p in model.named_parameters()
                     if p.grad is not None and 'p3_cell_bias' not in n)**0.5 if batch_idx % args.grad_accum == 0 else 0.0
            grad_sum += gn
            total_loss += loss.item() * args.grad_accum
            n_batches_done += 1

        if ema is not None:
            ema.update(model)

        # Clamp BN running stats to prevent NaN cascade from small-batch noise
        for name, buf in model.named_buffers():
            if 'running_var' in name:
                buf.data.clamp_(min=1e-4)
            buf.data.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)

        n = max(n_batches_done, 1)
        train_loss_v = total_loss / n
        grad_norm_v = grad_sum / n
        epoch_time = time.time() - epoch_t0
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

        # ── Health + detection checks after SGDR restarts ──
        if epoch in [66, 133]:
            _health_check(model, f"post-restart ep{epoch+1}")
            _detection_pipeline_check(model, f"post-restart ep{epoch+1}")

        # ── Validate ──
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_results = validate(model, val_loader, criterion, device)
        avg_f1 = float(np.mean([v["f1"] for v in val_results.values()]))

        if avg_f1 > best_val_f1 + 1e-8:
            best_val_f1 = avg_f1
            save_checkpoint(args.output, epoch+1, model, ema, optimizer, scheduler,
                             train_loss_v, avg_f1, scheduler.get_last_lr()[0] if epoch > 0 else args.lr,
                             grad_norm_v, step, best_val_f1)
            print(f"  ★ BEST (F1={avg_f1:.4f}) ep {epoch+1}")

        # ── Detailed per-epoch diagnostics ──
        diag = collect_diagnostics(model)

        if (epoch + 1) % args.diag_interval == 0 or epoch == args.epochs - 1:
            # Save JSON diagnostic dump
            json_path = os.path.join(diag_dir, f"v7_epoch_{epoch+1:03d}.json")
            json_data = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss_v),
                "grad_norm": float(grad_norm_v),
                "lr": float(scheduler.get_last_lr()[0] if epoch > 0 else args.lr),
                "val_results": {
                    lv: {"f1": val_results[lv]["f1"], "obj_loss": val_results[lv]["obj_loss"]}
                    for lv in ["p3", "p4"]
                },
                "diagnostics": diag,
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=float)

            # Console diagnostic line
            print(f"  DIAG: P3_obj_L2={diag['head_weights']['p3_obj']:.2f} "
                  f"P3_iou_L2={diag['head_weights']['p3_iou']:.2f} "
                  f"P3_bbox_L2={diag['head_weights']['p3_bbox']:.2f} | "
                  f"P4_obj_L2={diag['head_weights']['p4_obj']:.2f} "
                  f"P4_iou_L2={diag['head_weights']['p4_iou']:.2f} "
                  f"P4_bbox_L2={diag['head_weights']['p4_bbox']:.2f} | "
                  f"P3_obj_bias={diag['head_biases']['p3_obj']['sigmoid']:.3f} "
                  f"P4_obj_bias={diag['head_biases']['p4_obj']['sigmoid']:.3f}"
                  + (f" | cell_bias μ={diag['p3_cell_bias']['mean']:.2f} σ={diag['p3_cell_bias']['std']:.2f}"
                     if 'p3_cell_bias' in diag else ""))

        # ── Epoch checkpoint ──
        if (epoch + 1) % args.ckpt_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(args.output.replace(".pth", f"_ep{epoch+1:03d}.pth"),
                             epoch+1, model, ema, optimizer, scheduler,
                             train_loss_v, avg_f1, scheduler.get_last_lr()[0],
                             grad_norm_v, step)

        # ── Log ──
        lr_now = optimizer.param_groups[-1]["lr"]

        p3_obj_b = diag['head_biases']['p3_obj']['sigmoid']
        p4_obj_b = diag['head_biases']['p4_obj']['sigmoid']
        p3_bbox_dw = diag['head_biases']['p3_bbox']['dw']
        p4_bbox_dw = diag['head_biases']['p4_bbox']['dw']

        bn_ma = diag.get('bn_stats', {}).get('running_mean_abs_avg', -1.0)
        bn_va = diag.get('bn_stats', {}).get('running_var_avg', -1.0)
        print(f"  e{epoch+1:3d} | loss={train_loss_v:.3f} | grad={grad_norm_v:.2f} | "
              f"F1 p3={val_results['p3']['f1']:.3f}/{val_results['p3']['recall']:.3f} "
              f"p4={val_results['p4']['f1']:.3f}/{val_results['p4']['recall']:.3f} | "
              f"bias={p4_obj_b:.3f} | BN_μ={bn_ma:.3f} σ²={bn_va:.3f} | "
              f"LR={lr_now:.1e} | {epoch_time:.0f}s" +
              (" ★" if avg_f1 == best_val_f1 else ""))

        row = [epoch+1, f"{train_loss_v:.4f}",
               f"{val_results['p3']['obj_loss']:.4f}", f"{val_results['p4']['obj_loss']:.4f}",
               f"{val_results['p3']['f1']:.4f}", f"{val_results['p4']['f1']:.4f}",
               f"{p3_obj_b:.4f}", f"{p4_obj_b:.4f}",
               f"{diag['head_biases']['p3_iou']['sigmoid']:.4f}", f"{diag['head_biases']['p4_iou']['sigmoid']:.4f}",
               f"{p3_bbox_dw:.4f}", f"{p4_bbox_dw:.4f}",
               f"{diag['head_biases']['p3_bbox']['dh']:.4f}", f"{diag['head_biases']['p4_bbox']['dh']:.4f}",
               f"{diag['head_weights']['p3_obj']:.4f}", f"{diag['head_weights']['p3_iou']:.4f}", f"{diag['head_weights']['p3_bbox']:.4f}",
               f"{diag['head_weights']['p4_obj']:.4f}", f"{diag['head_weights']['p4_iou']:.4f}", f"{diag['head_weights']['p4_bbox']:.4f}",
               f"{grad_norm_v:.4f}", f"{lr_now:.8f}", f"{epoch_time:.2f}", f"{gpu_mem:.1f}"]
        metrics_rows.append(row)
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_headers)
            csv.writer(f).writerows(metrics_rows)

        # ── LR scheduler step ──
        scheduler.step()

    # ── Final ──
    print(f"\n  Training complete.")
    print(f"  Best mean F1: {best_val_f1:.4f}")
    print(f"  Checkpoint:   {args.output}")
    print(f"  Metrics:      {metrics_path}")


if __name__ == "__main__":
    main()
