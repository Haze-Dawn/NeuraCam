"""
FaceCNN v6.0 — Per-Level Detection Threshold Calibrator
=========================================================

Measures real detection-level precision/recall per FPN level, then finds
optimal per-level confidence thresholds that maximize detection F1 or
minimize FP rate at a target recall.

Unlike the previous threshold sweep (threshold_sweep.json) which matched at
the HEATMAP CELL level, this script:
  1. Runs the FULL inference pipeline (peak-finding, bbox decode, NMS)
  2. Tags each detection with its originating FPN level
  3. Evaluates detection precision/recall with IoU=0.5 matching
  4. Sweeps per-level thresholds independently to find optimal value per level

Why per-level thresholds matter:
  - P2 (240x320 grid): 100% cells fire at any threshold. Need HIGH threshold (~0.5-0.7)
    to suppress the 61M cell-level false positives.
  - P3 (120x160 grid): ~9,900 cells >0.05. Moderate threshold (~0.3-0.4).
  - P4 (60x80 grid): Only 16 cells >0.05. LOW threshold (~0.1-0.15) to catch more faces.

Usage:
  python scripts/per_level_threshold_calibrate.py \
      --model models/face_cnn_v6_best.pth \
      --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
      --output models/v6_posthoc/per_level_thresholds.json \
      --max-images 500

Output JSON contains:
  - per_level_metrics: precision, recall, F1 at each threshold per level
  - optimal_thresholds: best threshold per level (max F1)
  - recommended_thresholds: thresholds at target recall (90% of max per level)
  - combined_metrics: detection F1 when using optimal per-level thresholds
"""

import os, sys, json, argparse, time
from collections import defaultdict
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv.face_detector_cnn import FaceFCNv5
from src.cv.face_tracker import BoundingBox


# ── WIDER Face annotation parser ──

def parse_wider_annotations(root_dir, split="val"):
    annot_file = os.path.join(root_dir, "wider_face_split",
                               f"wider_face_{split}_bbx_gt.txt")
    img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
    gt = {}
    with open(annot_file) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        img_name = lines[i].strip()
        i += 1
        if i >= len(lines) or not img_name:
            break
        num_faces = int(lines[i].strip())
        i += 1
        img_path = os.path.join(img_dir, img_name)
        faces = []
        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            i += 1
            if len(parts) >= 4:
                x, y, w, h = map(int, parts[:4])
                if w > 5 and h > 5:
                    faces.append(BoundingBox(x=x, y=y, w=w, h=h))
        if faces and os.path.exists(img_path):
            gt[img_name] = {"path": img_path, "faces": faces}
    print(f"Loaded {len(gt)} images with "
          f"{sum(len(v['faces']) for v in gt.values())} face annotations ({split})")
    return gt


# ── IoU computation ──

def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    xo = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    yo = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = xo * yo
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-8)


# ── Per-level detection with peak-finding ──

@torch.no_grad()
def detect_per_level(model, frame, device, conf_thresholds):
    """Run full FPN inference, tag each detection with its FPN level.

    Args:
        model: FaceFCNv5 in eval mode
        frame: BGR numpy array (H, W, 3)
        device: torch device
        conf_thresholds: dict of {level: raw_logit_threshold} for peak finding.
                         If None, use default 0.3 sigmoid for all levels.

    Returns:
        list of (level, confidence, BoundingBox) tuples
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    out = model(tensor)

    levels = [
        ("p2", 2, out["p2_obj"][0, 0].cpu().numpy(),
         out["p2_bbox"][0].cpu().numpy()),
        ("p3", 4, out["p3_obj"][0, 0].cpu().numpy(),
         out["p3_bbox"][0].cpu().numpy()),
        ("p4", 8, out["p4_obj"][0, 0].cpu().numpy(),
         out["p4_bbox"][0].cpu().numpy()),
    ]

    all_dets = []
    for lname, stride, obj_map, bbox_map in levels:
        thresh_logit = float(np.log(conf_thresholds[lname] / (1.0 - conf_thresholds[lname])))

        # Peak finding: morphological dilation on raw logits, threshold in logit space
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(obj_map, kernel)
        peaks = (obj_map == dilated) & (obj_map > thresh_logit)

        if not peaks.any():
            continue

        ys, xs = np.where(peaks)
        for cy, cx in zip(ys, xs):
            conf = float(1.0 / (1.0 + np.exp(-obj_map[cy, cx])))
            dx = float(bbox_map[0, cy, cx])
            dy = float(bbox_map[1, cy, cx])
            dw = float(bbox_map[2, cy, cx])
            dh = float(bbox_map[3, cy, cx])

            box_cx = (cx + 0.5 + dx) * stride
            box_cy = (cy + 0.5 + dy) * stride
            box_w = float(np.exp(np.clip(dw, -2, 5))) * stride
            box_h = float(np.exp(np.clip(dh, -2, 5))) * stride

            if box_w < 5 or box_h < 5:
                continue
            x1 = int(max(0, box_cx - box_w / 2))
            y1 = int(max(0, box_cy - box_h / 2))
            bw = int(min(box_w, w - x1))
            bh = int(min(box_h, h - y1))
            if bw < 5 or bh < 5:
                continue

            all_dets.append((
                lname, conf,
                BoundingBox(x=x1, y=y1, w=bw, h=bh)
            ))

    # Cross-level NMS (greedy, keep highest confidence)
    all_dets.sort(key=lambda x: x[1], reverse=True)
    kept = []
    for det in all_dets:
        keep = True
        for k in kept:
            if compute_iou(det[2], k[2]) > 0.3:
                keep = False
                break
        if keep:
            kept.append(det)

    return kept


# ── Per-level threshold sweep ──

def sweep_per_level(gt_data, model, device, max_images=None):
    """Collect all detections at very low threshold, tagged by FPN level.
    Then simulate per-level threshold sweeps to find optimal values.
    """
    # Collect ALL detections at very low threshold
    low_thresh = {"p2": 0.001, "p3": 0.001, "p4": 0.001}  # sigmoid≈0.001 ≈ logit -6.9
    print(f"\nCollecting detections at ultra-low threshold (sigmoid≈0.007)...")
    print(f"  This captures all possible candidates for simulation.")

    all_detections = []  # (level, confidence, bbox, img_key)
    n_gt_total = 0
    images_processed = 0

    gt_items = list(gt_data.items())
    if max_images:
        gt_items = gt_items[:max_images]

    t0 = time.time()
    for img_name, ann in gt_items:
        img = cv2.imread(ann["path"])
        if img is None:
            continue
        dets = detect_per_level(model, img, device, low_thresh)
        for lv, conf, bbox in dets:
            all_detections.append((lv, conf, bbox, img_name))
        n_gt_total += len(ann["faces"])
        images_processed += 1
        if images_processed % 100 == 0:
            elapsed = time.time() - t0
            rate = images_processed / max(elapsed, 0.1)
            print(f"  {images_processed}/{len(gt_items)} images "
                  f"({rate:.1f} img/s, {len(all_detections)} detections so far)")

    elapsed = time.time() - t0
    print(f"  Collected {len(all_detections)} detections from "
          f"{images_processed} images in {elapsed:.1f}s "
          f"({images_processed/elapsed:.1f} img/s)")
    print(f"  Ground truth faces: {n_gt_total}")

    # ── Simulate per-level threshold sweeps ──
    p2_thresholds = [round(v, 2) for v in np.arange(0.05, 0.75, 0.05)]
    p3_thresholds = [round(v, 2) for v in np.arange(0.05, 0.65, 0.05)]
    p4_thresholds = [round(v, 2) for v in np.arange(0.03, 0.55, 0.04)]

    # For each level independently, sweep while keeping others at ultra-low
    # This gives us the marginal contribution of each level's threshold
    results = {}
    for target_level in ["p2", "p3", "p4"]:
        thresholds = {"p2": p2_thresholds, "p3": p3_thresholds,
                       "p4": p4_thresholds}[target_level]
        level_results = []

        # Other levels stay at ultra-low (include everything)
        other_thresholds = {lv: 0.001 for lv in ["p2", "p3", "p4"]}

        for t in thresholds:
            other_thresholds[target_level] = t
            metrics = evaluate_with_thresholds(
                all_detections, gt_data, other_thresholds, n_gt_total)
            metrics["threshold"] = t
            level_results.append(metrics)

        # Find best threshold
        best = max(level_results, key=lambda x: x["f1"])
        results[target_level] = {
            "sweep": level_results,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_tp": best["tp"],
            "best_fp": best["fp"],
        }

        print(f"\n  {target_level.upper()} sweep: best threshold={best['threshold']:.2f} "
              f"(P={best['precision']:.4f}, R={best['recall']:.4f}, "
              f"F1={best['f1']:.4f}, FP={best['fp']})")

    # ── Combined: use best per-level thresholds together ──
    combined_thresholds = {
        lv: results[lv]["best_threshold"] for lv in ["p2", "p3", "p4"]
    }
    combined_metrics = evaluate_with_thresholds(
        all_detections, gt_data, combined_thresholds, n_gt_total)
    combined_metrics["thresholds"] = combined_thresholds

    print(f"\n  Combined (optimal per-level): "
          f"P={combined_metrics['precision']:.4f}, "
          f"R={combined_metrics['recall']:.4f}, "
          f"F1={combined_metrics['f1']:.4f}, "
          f"TP={combined_metrics['tp']}, FP={combined_metrics['fp']}")

    # ── Target-recall thresholds (90% of max recall per level) ──
    recommended = {}
    for lv in ["p2", "p3", "p4"]:
        best_recall = results[lv]["best_recall"]
        target_recall = best_recall * 0.90
        # Find highest threshold that achieves >= target_recall
        best_t = results[lv]["best_threshold"]
        for entry in results[lv]["sweep"]:
            if entry["recall"] >= target_recall:
                best_t = max(best_t, entry["threshold"])
        recommended[lv] = best_t

    combined_rec = evaluate_with_thresholds(
        all_detections, gt_data, recommended, n_gt_total)
    results["recommended"] = {
        "thresholds": recommended,
        "metrics": combined_rec,
    }

    print(f"\n  Recommended (90% recall): {recommended}")
    print(f"    P={combined_rec['precision']:.4f}, "
          f"R={combined_rec['recall']:.4f}, "
          f"F1={combined_rec['f1']:.4f}")

    results["combined_optimal"] = combined_metrics
    results["n_ground_truth"] = n_gt_total
    results["n_detections_collected"] = len(all_detections)
    results["images_processed"] = images_processed

    return results


def evaluate_with_thresholds(all_detections, gt_data, thresholds, n_gt_total):
    """Evaluate detection metrics with given per-level thresholds.
    Uses greedy IoU=0.5 matching (one GT per detection).
    """
    # Filter detections by per-level threshold
    filtered = [(lv, conf, bbox, img_key)
                for lv, conf, bbox, img_key in all_detections
                if conf >= thresholds.get(lv, 0.3)]

    # Sort by confidence descending for greedy matching
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Group ground truth by image
    gt_by_image = {k: list(v["faces"]) for k, v in gt_data.items()}

    tp = 0
    fp = 0
    for lv, conf, det_bbox, img_key in filtered:
        gt_boxes = gt_by_image.get(img_key, [])
        if not gt_boxes:
            fp += 1
            continue

        best_iou = 0
        best_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if gt_box is None:  # already matched
                continue
            iou = compute_iou(det_bbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= 0.5 and best_idx >= 0:
            tp += 1
            gt_boxes[best_idx] = None  # mark as matched
        else:
            fp += 1

    fn = max(0, n_gt_total - tp)

    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt_total, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Per-Level Detection Threshold Calibrator")
    parser.add_argument("--model", required=True,
                        help="Path to FaceCNN checkpoint")
    parser.add_argument("--data", required=True,
                        help="Path to WIDER Face data directory")
    parser.add_argument("--output", default="models/v6_posthoc/per_level_thresholds.json",
                        help="Output JSON path")
    parser.add_argument("--split", default="val",
                        help="WIDER split to evaluate on")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images for faster sweep (default: all)")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA weights from checkpoint (recommended)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    model = FaceFCNv5().to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    sd = ckpt.get('ema_state_dict' if args.ema else 'model_state_dict',
                   ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {total_p:,} params")
    print(f"Using {'EMA' if args.ema and 'ema_state_dict' in ckpt else 'raw'} weights")

    # ── Load ground truth ──
    gt = parse_wider_annotations(args.data, args.split)

    # ── Sweep per-level thresholds ──
    results = sweep_per_level(gt, model, device, args.max_images)

    # ── Save ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"  P2 optimal threshold: {results['p2']['best_threshold']:.2f}")
    print(f"  P3 optimal threshold: {results['p3']['best_threshold']:.2f}")
    print(f"  P4 optimal threshold: {results['p4']['best_threshold']:.2f}")
    print(f"  Combined F1: {results['combined_optimal']['f1']:.4f}")
    print(f"  Recommended thresholds: {results['recommended']['thresholds']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
