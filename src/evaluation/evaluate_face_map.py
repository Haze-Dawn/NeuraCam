"""WIDER Face mAP evaluation — compares FaceCNN detections against ground truth.
Computes precision-recall curve and mean Average Precision at IoU=0.5.

Usage: PYTHONPATH="." python src/evaluation/evaluate_face_map.py
"""
import os, sys, json, glob
import cv2
import numpy as np
from src.cv.face_detector_cnn import FaceCNN
from src.control.kalman import BoundingBox


def parse_wider_annotations(root_dir, split="val"):
    """Parse WIDER Face ground truth. Returns dict: image_rel_path → [BoundingBox]."""
    annot_file = os.path.join(root_dir, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")
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
    print(f"Loaded {len(gt)} images with {sum(len(v['faces']) for v in gt.values())} face annotations")
    return gt


def compute_iou(a, b):
    xo = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    yo = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = xo * yo
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def evaluate_map(gt_data, detector, conf_thresholds=None):
    if conf_thresholds is None:
        conf_thresholds = np.linspace(0.05, 0.95, 19)

    all_dets = []  # (confidence, is_match) for every detection
    n_gt = 0

    for img_name, ann in gt_data.items():
        img = cv2.imread(ann["path"])
        if img is None:
            continue
        gt_boxes = ann["faces"]
        n_gt += len(gt_boxes)

        faces = detector.detect(img)
        matched_gt = set()

        # Sort detections by confidence descending
        faces.sort(key=lambda f: f.confidence, reverse=True)

        for det in faces:
            best_iou = 0
            best_idx = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(det.bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            is_match = 1.0 if best_iou >= 0.5 and best_idx >= 0 else 0.0
            all_dets.append((det.confidence, is_match))
            if is_match:
                matched_gt.add(best_idx)

    all_dets.sort(key=lambda x: x[0], reverse=True)

    # Compute precision-recall curve
    precisions = []
    recalls = []
    tp = 0
    fp = 0
    for conf, is_match in all_dets:
        if is_match:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(n_gt, 1))

    # mAP = area under PR curve (trapezoidal integration)
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i - 1])

    # Per-threshold metrics
    threshold_metrics = []
    for t in conf_thresholds:
        tp_t = sum(1 for c, m in all_dets if c >= t and m)
        fp_t = sum(1 for c, m in all_dets if c >= t and not m)
        fn_t = n_gt - tp_t
        prec_t = tp_t / max(tp_t + fp_t, 1)
        rec_t = tp_t / max(n_gt, 1)
        threshold_metrics.append({
            "threshold": round(t, 2),
            "precision": round(prec_t, 4),
            "recall": round(rec_t, 4),
            "tp": tp_t, "fp": fp_t, "fn": fn_t,
        })

    return {
        "mAP": round(ap, 4),
        "n_ground_truth": n_gt,
        "n_detections": len(all_dets),
        "threshold_metrics": threshold_metrics,
        "precision_interp": [round(p, 4) for p in precisions],
        "recall_interp": [round(r, 4) for r in recalls],
        "best_f1": round(max(
            2 * t["precision"] * t["recall"] / max(t["precision"] + t["recall"], 1e-8)
            for t in threshold_metrics
        ), 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/face/widerface")
    parser.add_argument("--model", default="models/face_cnn_best.pth")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for quick test")
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    detector = FaceCNN(model_path=args.model, confidence_threshold=0.05)

    print(f"Loading WIDER Face {args.split} annotations...")
    gt = parse_wider_annotations(args.data, args.split)

    if args.max_images:
        keys = list(gt.keys())[:args.max_images]
        gt = {k: gt[k] for k in keys}
        print(f"  Limited to {len(gt)} images")

    print("Running evaluation...")
    results = evaluate_map(gt, detector)

    out_dir = os.path.join(args.output, "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wider_face_map.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"mAP @ IoU=0.5: {results['mAP']:.4f}")
    print(f"Best F1:       {results['best_f1']:.4f}")
    print(f"GT faces:      {results['n_ground_truth']}")
    print(f"Detections:    {results['n_detections']}")
    print(f"\nPer-threshold breakdown:")
    for t in results["threshold_metrics"][::3]:
        print(f"  thresh={t['threshold']:.2f}: P={t['precision']:.3f} R={t['recall']:.3f} TP={t['tp']} FP={t['fp']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
