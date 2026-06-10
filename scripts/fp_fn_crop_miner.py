"""
FaceCNN v6.0 — Targeted FP/FN Mining Diagnostic
=================================================

Runs the full detection pipeline on WIDER val, records every detection as
TP (IoU≥0.5 match) or FP (no match), and every ground truth face as FN
(no detection matched). Saves actual image crops of FPs and FNs for
targeted fine-tuning — traning the model specifically on what it gets wrong.

Output:
  - fp_crops.pt: Bounding-box crops of false-positive detections (top-K by confidence)
  - fn_crops.pt: Bounding-box crops of missed faces (false negatives)
  - per_image_failures.json: Per-image breakdown of TP/FP/FN counts

Usage:
  python scripts/fp_fn_crop_miner.py \
      --data "/path/to/Data" \
      --model models/face_cnn_v6_best.pth \
      --output-dir models/v6_posthoc/fp_fn_crops \
      --max-images 500
"""

import os, sys, json, argparse, time
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv.face_detector_cnn import FaceFCNv5
from src.cv.face_tracker import BoundingBox


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
                if w > 20 and h > 20:  # Minimum face size
                    faces.append(BoundingBox(x=x, y=y, w=w, h=h))
        if faces and os.path.exists(img_path):
            gt[img_name] = {"path": img_path, "faces": faces}
    return gt


def compute_iou(a, b):
    xo = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    yo = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = xo * yo
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-8)


@torch.no_grad()
def detect_with_levels(model, frame, device):
    """Run full detection pipeline, return (level, confidence, BoundingBox) tuples."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    out = model(tensor)

    levels = [
        ("p2", 2, out["p2_obj"][0, 0].cpu().numpy(), out["p2_bbox"][0].cpu().numpy()),
        ("p3", 4, out["p3_obj"][0, 0].cpu().numpy(), out["p3_bbox"][0].cpu().numpy()),
        ("p4", 8, out["p4_obj"][0, 0].cpu().numpy(), out["p4_bbox"][0].cpu().numpy()),
    ]

    sigmoid_thresh = 0.001
    logit_thresh = float(np.log(sigmoid_thresh / (1.0 - sigmoid_thresh)))

    all_dets = []
    for lname, stride, obj_map, bbox_map in levels:
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(obj_map, kernel)
        peaks = (obj_map == dilated) & (obj_map > logit_thresh)
        if not peaks.any():
            continue

        ys, xs = np.where(peaks)
        for cy, cx in zip(ys, xs):
            conf = float(1.0 / (1.0 + np.exp(-obj_map[cy, cx])))
            dx, dy = float(bbox_map[0, cy, cx]), float(bbox_map[1, cy, cx])
            dw, dh = float(bbox_map[2, cy, cx]), float(bbox_map[3, cy, cx])

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
            all_dets.append((lname, conf, BoundingBox(x=x1, y=y1, w=bw, h=bh)))

    # NMS
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


def mine_fp_fn(gt_data, model, device, output_dir, max_images=None, max_crops=500):
    """Run detection on all images, classify each detection as TP/FP,
    and each GT face as TP/FN at IoU=0.5. Extract crops of FPs and FNs."""
    os.makedirs(output_dir, exist_ok=True)

    fp_crops = []   # (confidence, crop_tensor, level)
    fn_crops = []   # (gt_box_size, crop_tensor)
    tp_count = 0
    fp_count = 0
    fn_count = 0
    n_gt_total = 0
    images_processed = 0

    per_image = []

    gt_items = list(gt_data.items())
    if max_images:
        gt_items = gt_items[:max_images]

    t0 = time.time()
    for img_name, ann in gt_items:
        img = cv2.imread(ann["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = ann["faces"]
        n_gt_total += len(gt_boxes)

        dets = detect_with_levels(model, img, device)
        matched_gt = set()

        img_tp = 0
        img_fp = 0
        img_fn = 0

        for lv, conf, det_box in dets:
            best_iou = 0
            best_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= 0.5 and best_idx >= 0:
                tp_count += 1
                img_tp += 1
                matched_gt.add(best_idx)
            else:
                fp_count += 1
                img_fp += 1
                # Extract FP crop from the detection box
                x1, y1 = max(0, det_box.x), max(0, det_box.y)
                x2, y2 = min(w, det_box.x + det_box.w), min(h, det_box.y + det_box.h)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    crop = img[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_t = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
                    fp_crops.append((conf, crop_t, lv))

        # FNs: GT faces not matched
        for j, gt_box in enumerate(gt_boxes):
            if j not in matched_gt:
                fn_count += 1
                img_fn += 1
                # Extract FN crop from GT box
                x1, y1 = max(0, gt_box.x), max(0, gt_box.y)
                x2, y2 = min(w, gt_box.x + gt_box.w), min(h, gt_box.y + gt_box.h)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    crop = img[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_t = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
                    fn_crops.append(((gt_box.w * gt_box.h), crop_t))

        per_image.append({
            "image": img_name,
            "gt_faces": len(gt_boxes),
            "detections": len(dets),
            "tp": img_tp, "fp": img_fp, "fn": img_fn,
        })
        images_processed += 1

        if images_processed % 50 == 0:
            elapsed = time.time() - t0
            rate = images_processed / max(elapsed, 0.1)
            print(f"  {images_processed}/{len(gt_items)} images "
                  f"({rate:.1f} img/s) | TP={tp_count} FP={fp_count} FN={fn_count}")

    elapsed = time.time() - t0
    print(f"\n  Processed {images_processed} images in {elapsed:.1f}s "
          f"({images_processed/elapsed:.1f} img/s)")
    print(f"  TP={tp_count} | FP={fp_count} | FN={fn_count} | GT={n_gt_total}")
    if n_gt_total > 0:
        print(f"  Detection recall: {tp_count/n_gt_total:.4f} "
              f"({n_gt_total - fn_count}/{n_gt_total})")

    # Sort FPs by confidence (most confidently-wrong first)
    fp_crops.sort(key=lambda x: x[0], reverse=True)
    fp_crops = fp_crops[:max_crops]

    # Sort FNs by box size (largest missed faces first — most important)
    fn_crops.sort(key=lambda x: x[0], reverse=True)
    fn_crops = fn_crops[:max_crops]

    # Save FP crops (these are the hard negatives for targeted fine-tuning)
    fp_data = {
        "crops": [(conf, crop, lv) for conf, crop, lv in fp_crops],
        "count": len(fp_crops),
        "description": "False-positive detection crops, sorted by confidence. "
                       "These are image regions the model thinks are faces but "
                       "don't match any GT face (IoU<0.5). Top entries are the "
                       "most confidently-wrong predictions — prime hard negatives.",
    }
    fp_path = os.path.join(output_dir, "fp_targeted_crops.pt")
    torch.save(fp_data, fp_path)
    print(f"\n  FP crops saved: {fp_path} ({len(fp_crops)} crops)")

    # Save FN crops (hard positives — faces the model misses)
    fn_data = {
        "crops": [(size, crop) for size, crop in fn_crops],
        "count": len(fn_crops),
        "description": "False-negative GT face crops. These are faces the model "
                       "failed to detect (no detection matched at IoU>=0.5). "
                       "Sorted by face size (largest first). These are prime "
                       "candidates for positive reinforcement fine-tuning.",
    }
    fn_path = os.path.join(output_dir, "fn_targeted_crops.pt")
    torch.save(fn_data, fn_path)
    print(f"  FN crops saved: {fn_path} ({len(fn_crops)} crops)")

    # Save per-image stats
    stats_path = os.path.join(output_dir, "per_image_failures.json")
    summary = {
        "n_images": images_processed,
        "n_gt_total": n_gt_total,
        "tp_total": tp_count,
        "fp_total": fp_count,
        "fn_total": fn_count,
        "detection_recall": round(tp_count / max(n_gt_total, 1), 4),
        "per_image": per_image,
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Per-image stats saved: {stats_path}")

    return fp_path, fn_path, stats_path


def main():
    parser = argparse.ArgumentParser(
        description="Targeted FP/FN Crop Miner for Fine-Tuning")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="models/v6_posthoc/fp_fn_crops")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-crops", type=int, default=500,
                        help="Max FP/FN crops to save")
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FaceFCNv5().to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    sd = ckpt.get('ema_state_dict' if args.ema else 'model_state_dict',
                   ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded model: {args.model}")
    print(f"Using {'EMA' if args.ema and 'ema_state_dict' in ckpt else 'raw'} weights")

    gt = parse_wider_annotations(args.data, "val")
    print(f"Loaded {len(gt)} annotated images ({sum(len(v['faces']) for v in gt.values())} faces)")

    print(f"\nMining FP/FN crops from {args.max_images or len(gt)} images...")
    fp_path, fn_path, stats_path = mine_fp_fn(
        gt, model, device, args.output_dir, args.max_images, args.max_crops)

    print(f"\n{'='*60}")
    print(f"Targeted mining complete.")
    print(f"  FP crops (hard negatives): {fp_path}")
    print(f"  FN crops (hard positives): {fn_path}")
    print(f"  Per-image stats: {stats_path}")
    print(f"\nNext: use these crops for targeted fine-tuning:")
    print(f"  python -m src.training.train_v6 --resume models/face_cnn_v6_best.pth \\")
    print(f"      --targeted-fp models/v6_posthoc/fp_fn_crops/fp_targeted_crops.pt \\")
    print(f"      --targeted-fn models/v6_posthoc/fp_fn_crops/fn_targeted_crops.pt \\")
    print(f"      --resume-lr-override 5e-4 --flat-lr --epochs 75 ...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
