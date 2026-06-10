"""
FaceCNN v6.0 — Post-Hoc Improvement Suite (Epoch 56)
=====================================================

Phase A (minutes, zero retraining):
  1. EMA weights extraction for inference
  2. Per-level confidence threshold sweep on WIDER val set
  3. Hard-negative mining diagnostic

Phase B (1.5 hours, uses epoch 56 checkpoint):
  4. Hard-negative fine-tuning (15 epochs, --mine-retrain)

Usage:
  python scripts/v6_posthoc_improve.py --data /path/to/Data --checkpoint models/face_cnn_v6_best.pth
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv.face_detector_cnn import FaceFCNv5
from src.training.train_face_cnn import WiderFaceFPNDataset
from src.training.train_v6 import hard_negative_mining


# ============================================================================
# STEP 1: EMA Weight Extraction
# ============================================================================

def extract_ema_weights(checkpoint_path, output_path):
    """Extract EMA model weights from training checkpoint for inference."""
    print("\n" + "=" * 70)
    print("  STEP 1: EMA Weight Extraction")
    print("=" * 70)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    epoch = ckpt.get('epoch', '?')
    val_f1 = ckpt.get('val_f1', 0.0)
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Checkpoint val F1: {val_f1:.4f}")

    if 'ema_state_dict' not in ckpt:
        print("  ERROR: No ema_state_dict found in checkpoint.")
        return None

    ema_sd = ckpt['ema_state_dict']
    model_sd = ckpt['model_state_dict']

    # Verify EMA has all required keys
    missing = set(model_sd.keys()) - set(ema_sd.keys())
    if missing:
        print(f"  WARNING: EMA missing {len(missing)} keys, filling from model")
        for k in missing:
            ema_sd[k] = model_sd[k]

    torch.save(ema_sd, output_path)
    print(f"  EMA weights saved to: {output_path}")
    print(f"  Tensors: {len(ema_sd)}")
    return output_path


# ============================================================================
# STEP 2: Per-Level Confidence Threshold Sweep
# ============================================================================

@torch.no_grad()
def threshold_sweep(model_path, data_root, device, output_path):
    """Sweep confidence thresholds per FPN level on WIDER val set."""
    print("\n" + "=" * 70)
    print("  STEP 2: Per-Level Confidence Threshold Sweep")
    print("=" * 70)

    model = FaceFCNv5().to(device)
    sd = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()

    val_dataset = WiderFaceFPNDataset(
        data_root, "val", target_h=480, target_w=640, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                             num_workers=1, pin_memory=True)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    levels = ['p2', 'p3', 'p4']

    results = {lv: {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds}
               for lv in levels}

    print(f"  Evaluating {len(thresholds)} thresholds on {len(val_dataset)} images...")
    n_batches = min(100, len(val_loader))

    for batch_idx, batch in enumerate(tqdm(val_loader, total=n_batches, desc="Thresh sweep")):
        if batch_idx >= n_batches:
            break
        patches, hm_p2, _, hm_p3, _, hm_p4, _ = batch
        patches = patches.to(device)
        outputs = model(patches)

        for lv in levels:
            obj = torch.sigmoid(outputs[f'{lv}_obj']).cpu()
            gt_map = {'p2': hm_p2, 'p3': hm_p3, 'p4': hm_p4}
            gt = gt_map[lv]
            gt = gt.squeeze(1)

            for t in thresholds:
                pred = (obj.squeeze(1) > t).float()
                gt_bin = (gt > 0.5).float()
                results[lv][t]['tp'] += (pred * gt_bin).sum().item()
                results[lv][t]['fp'] += (pred * (1 - gt_bin)).sum().item()
                results[lv][t]['fn'] += ((1 - pred) * gt_bin).sum().item()

    best = {}
    print(f"\n  {'Level':<6} {'Thresh':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'TP':<8} {'FP':<8}")
    print(f"  {'-'*54}")
    for lv in levels:
        best_t = None
        best_f1 = 0
        for t in thresholds:
            r = results[lv][t]
            p = r['tp'] / max(r['tp'] + r['fp'], 1)
            rec = r['tp'] / max(r['tp'] + r['fn'], 1)
            f1 = 2 * p * rec / max(p + rec, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        r = results[lv][best_t]
        p = r['tp'] / max(r['tp'] + r['fp'], 1)
        rec = r['tp'] / max(r['tp'] + r['fn'], 1)
        best[lv] = {'threshold': best_t, 'f1': best_f1, 'precision': p, 'recall': rec}
        print(f"  {lv.upper():<6} {best_t:<8.2f} {p:<8.4f} {rec:<8.4f} {best_f1:<8.4f} {r['tp']:<8.0f} {r['fp']:<8.0f}")

    best['recommendations'] = {
        'p4': best.get('p4', {}).get('threshold', 0.3),
        'p3': best.get('p3', {}).get('threshold', 0.3),
        'p2': best.get('p2', {}).get('threshold', 0.3),
    }

    with open(output_path, 'w') as f:
        json.dump({'best_per_level': best, 'full_sweep': results}, f, indent=2, default=float)
    print(f"\n  Results saved to: {output_path}")
    return best


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FaceCNN v6.0 Post-Hoc Improvements")
    parser.add_argument("--data", required=True, help="Path to WIDER Face data dir")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to epoch 56 checkpoint (face_cnn_v6_best.pth)")
    parser.add_argument("--output-dir", default="models/v6_posthoc",
                        help="Output directory for all artifacts")
    parser.add_argument("--steps", default="1,2,3",
                        help="Comma-separated steps to run (1=EMA, 2=thresh, 3=mining)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    steps = [int(s.strip()) for s in args.steps.split(",")]

    # Step 1: EMA Export
    if 1 in steps:
        ema_path = os.path.join(args.output_dir, "face_cnn_v6_ema.pth")
        extract_ema_weights(args.checkpoint, ema_path)

    # Step 2: Threshold Sweep
    if 2 in steps:
        ema_path = os.path.join(args.output_dir, "face_cnn_v6_ema.pth")
        model_path = ema_path if os.path.exists(ema_path) else args.checkpoint
        thresh_path = os.path.join(args.output_dir, "threshold_sweep.json")
        threshold_sweep(model_path, args.data, device, thresh_path)

    # Step 3: Mining Diagnostic
    if 3 in steps:
        print("\n" + "=" * 70)
        print("  STEP 3: Hard-Negative Mining Diagnostic")
        print("=" * 70)

        model = FaceFCNv5().to(device)
        sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
        ema_sd = sd.get('ema_state_dict', sd.get('model_state_dict'))
        model.load_state_dict(ema_sd, strict=False)
        model.eval()

        print(f"  Mining on {args.data}/WIDER_train...")
        hard_negs, n_mined = hard_negative_mining(
            model, args.data, device, num_images=2000, conf_thresh=0.3, top_k=96)
        print(f"  Mining complete: {n_mined} hard negatives found.")

        if hard_negs:
            mine_path = os.path.join(args.output_dir, "hard_negatives.pt")
            torch.save({'negatives': hard_negs, 'count': n_mined}, mine_path)
            print(f"  Hard negatives saved to: {mine_path}")

    print("\n" + "=" * 70)
    print("  Post-hoc improvement suite complete.")
    print(f"  Artifacts in: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
