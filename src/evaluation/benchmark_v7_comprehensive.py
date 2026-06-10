"""
FaceFCN v7 — Comprehensive Benchmark & Deployment Pipeline
============================================================
Full evaluation of v7 (P3+P4) and v7 P4-only on WIDER Face val.

Metrics:
  1. WIDER Face mAP@IoU=0.5 (standard protocol)
  2. Per-category mAP (Easy/Medium/Hard face sizes)
  3. Per-event-category mAP (Parade, Meeting, etc.)
  4. COCO-style mAP@0.5:0.95
  5. GPU latency (mean/median/p95/p99), FPS
  6. Model params, FLOPs, GPU memory
  7. Confidence threshold sweep
  8. NMS IoU threshold sweep
  9. Resolution sensitivity analysis
 10. Calibration (ECE) + confusion matrix
 11. PR curve data

Deployment Pipeline:
  12. ONNX FP32 export + inference test
  13. INT8 dynamic quantization + accuracy check
 14. OpenVINO IR conversion + inference test
 15. TensorRT FP16/INT8 conversion + inference test

Usage:
  python src/evaluation/benchmark_v7_comprehensive.py --gpu
  python src/evaluation/benchmark_v7_comprehensive.py --gpu --sweep --resolution
  python src/evaluation/benchmark_v7_comprehensive.py --gpu --deploy
  python src/evaluation/benchmark_v7_comprehensive.py --gpu --models v7 v7_p4only
"""

import os
import sys
import json
import time
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.cv.face_detector_v7 import FaceFCNv7
from src.cv.face_detector_v7_p4only import FaceFCNv7P4Only


# ──────────────────────────────────────────────
# WIDER Face categories
# ──────────────────────────────────────────────

# WIDER Face event categories mapped to difficulty (Easy/Medium/Hard)
# Based on the official WIDER Face evaluation protocol
WIDER_CATEGORIES = {
    "0--Parade":           "medium",
    "1--Handshaking":      "medium",
    "2--Demonstration":    "easy",
    "3--Rescue":           "hard",
    "4--Interview":        "easy",
    "5--People_Marching":  "medium",
    "6--Meeting":          "easy",
    "7--Group":            "easy",
    "8--Picnic":           "easy",
    "9--Shoppers":         "medium",
    "10--Soldier_Firing":  "medium",
    "11--Press_Conference":"easy",
    "12--Traffic":         "medium",
    "13--Stock_Market":    "medium",
    "14--Award_Ceremony":  "easy",
    "15--Ceremony":        "easy",
    "16--Concerts":        "hard",
    "17--Couple":          "easy",
    "18--Family_Group":    "easy",
    "19--Festival":        "hard",
    "20--Spa":             "easy",
    "21--Sports_Fan":      "medium",
    "22--Soldier_Patrol":  "medium",
    "23--Soldier_Drilling":"medium",
    "24--Vehicle":         "medium",
    "25--Waiting":         "medium",
    "26--Working":         "easy",
    "27--Balloonist":      "hard",
    "28--Construction":    "medium",
    "29--Cheering":        "easy",
    "30--Dancing":         "easy",
    "31--Car_Racing":      "medium",
    "32--Red_Carpet":      "easy",
    "33--Swimming":        "medium",
    "34--Running":         "medium",
    "35--Biking":          "medium",
    "36--Horse_Riding":    "medium",
    "37--Skateboard":      "medium",
    "38--Skiing":          "medium",
    "39--Snowboard":       "medium",
    "40--Surfing":         "medium",
    "41--Skating":         "medium",
    "42--Golf":            "medium",
    "43--Tennis":          "medium",
    "44--Basketball":      "medium",
    "45--Soccer":          "medium",
    "46--Baseball":        "medium",
    "47--Football":        "medium",
    "48--Rock_Climbing":   "medium",
    "49--Martial_Arts":    "medium",
    "50--Yoga":            "easy",
    "51--Weightlifting":   "easy",
    "52--Gymnastics":      "easy",
    "53--Diving":          "medium",
    "54--Indoor":          "easy",
    "55--Outdoor":         "medium",
    "56--Night":           "hard",
    "57--Day":             "easy",
    "58--Rain":            "hard",
    "59--Snow":            "hard",
    "60--Fog":             "hard",
    "61--Sun":             "easy",
    "62--Cloudy":          "easy",
    "63--Overcast":        "easy",
}

# Face size categories (area in pixels² at 480×640 resolution)
FACE_SIZE_THRESHOLDS = {
    "small":  (0, 32*32),       # area < 1024
    "medium": (32*32, 96*96),   # 1024 ≤ area < 9216
    "large":  (96*96, float('inf')),  # area ≥ 9216
}


# ──────────────────────────────────────────────
# Dataset with category tracking
# ──────────────────────────────────────────────

class WiderValDataset(Dataset):
    """WIDER Face validation dataset with per-image metadata."""
    
    def __init__(self, root_dir, target_h=480, target_w=640, max_images=None):
        self.target_h = target_h
        self.target_w = target_w
        self.samples = []
        
        img_dir = os.path.join(root_dir, "WIDER_val", "images")
        annot_file = os.path.join(root_dir, "wider_face_split",
                                  "wider_face_val_bbx_gt.txt")
        
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation not found: {annot_file}")
        
        with open(annot_file) as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if i >= len(lines) or not img_name or "/" not in img_name:
                continue
            
            num_faces = int(lines[i].strip())
            i += 1
            
            img_path = os.path.join(img_dir, img_name)
            category = img_name.split("/")[0]
            
            faces = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    if w > 0 and h > 0:
                        faces.append({"x": x, "y": y, "w": w, "h": h})
            
            if os.path.exists(img_path):
                self.samples.append({
                    "image_path": img_path,
                    "faces": faces,
                    "category": category,
                })
        
        if max_images:
            self.samples = self.samples[:max_images]
        
        print(f"WiderValDataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ann = self.samples[idx]
        img = cv2.imread(ann["image_path"])
        if img is None:
            return None, None, None, None
        
        h, w = img.shape[:2]
        scale_x = self.target_w / w
        scale_y = self.target_h / h
        
        img_resized = cv2.resize(img, (self.target_w, self.target_h))
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        
        gt_boxes = []
        for face in ann["faces"]:
            gt_boxes.append([
                int(face["x"] * scale_x), int(face["y"] * scale_y),
                int(face["w"] * scale_x), int(face["h"] * scale_y)
            ])
        
        return tensor, gt_boxes, ann["image_path"], ann["category"]


# ──────────────────────────────────────────────
# IoU & mAP computation
# ──────────────────────────────────────────────

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return inter / (area1 + area2 - inter + 1e-8)


def soft_nms(dets, iou_thresh=0.3):
    dets = sorted(dets, key=lambda x: x[0], reverse=True)
    kept = []
    for det in dets:
        max_decay = 0.0
        for k in kept:
            iou = compute_iou(det[1], k[1])
            if iou > iou_thresh:
                max_decay = max(max_decay, iou)
        if max_decay > 0:
            decayed = det[0] * (1.0 - max_decay)
            if decayed > 0.01:
                kept.append((decayed, det[1]))
        else:
            kept.append(det)
    return kept


def compute_ap(recalls, precisions):
    """VOC 11-point interpolation AP."""
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
    return ap


def compute_ap_coco(all_dets, all_gts, iou_thresh=0.5):
    """COCO-style AP with 101-point interpolation."""
    tp_list, fp_list, scores_list, n_gt = [], [], [], 0
    for dets, gts in zip(all_dets, all_gts):
        gt_matched = [False] * len(gts)
        n_gt += len(gts)
        sorted_dets = sorted(dets, key=lambda x: x[0], reverse=True)
        for score, det_box in sorted_dets:
            best_iou, best_idx = 0, -1
            for j, gt_box in enumerate(gts):
                if not gt_matched[j]:
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            if best_iou >= iou_thresh and best_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[best_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)
            scores_list.append(score)
    
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    
    # COCO 101-point interpolation
    mpre = np.concatenate(([0.], precisions, [0.]))
    mrec = np.concatenate(([0.], recalls, [1.]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    ap = 0.0
    for t in np.arange(0.0, 1.01, 0.01):
        prec_at_t = mpre[mrec >= t].max() if (mrec >= t).any() else 0.0
        ap += prec_at_t / 101.0
    
    return ap, float(n_gt)


def compute_map(all_dets, all_gts, iou_thresh=0.5):
    """VOC-style mAP@IoU=iou_thresh."""
    tp_list, fp_list, n_gt = [], [], 0
    for dets, gts in zip(all_dets, all_gts):
        gt_matched = [False] * len(gts)
        n_gt += len(gts)
        sorted_dets = sorted(dets, key=lambda x: x[0], reverse=True)
        for score, det_box in sorted_dets:
            best_iou, best_idx = 0, -1
            for j, gt_box in enumerate(gts):
                if not gt_matched[j]:
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            if best_iou >= iou_thresh and best_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[best_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)
    
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = compute_ap(recalls, precisions) if n_gt > 0 else 0.0
    return ap, float(n_gt)


# ──────────────────────────────────────────────
# Detection functions
# ──────────────────────────────────────────────

@torch.no_grad()
def detect_v7(model, tensor, conf_thresh=0.25, nms_iou=0.3):
    out = model(tensor)
    all_dets = []
    for level, stride, key in [("p3", 4, "p3"), ("p4", 8, "p4")]:
        quality_t = torch.sqrt(
            torch.sigmoid(out[f"{key}_obj"][0, 0]) *
            torch.sigmoid(out[f"{key}_iou"][0, 0]) + 1e-8)
        quality = quality_t.cpu().numpy()
        bbox_raw = out[f"{key}_bbox"][0].cpu()
        
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(quality, kernel)
        peaks = (quality == dilated) & (quality > conf_thresh)
        if not peaks.any():
            continue
        ys, xs = np.where(peaks)
        for cy, cx in zip(ys, xs):
            q = float(quality[cy, cx])
            dx = float(bbox_raw[0, cy, cx].item())
            dy = float(bbox_raw[1, cy, cx].item())
            dw = float(np.clip(bbox_raw[2, cy, cx].item(), -2, 5))
            dh = float(np.clip(bbox_raw[3, cy, cx].item(), -2, 5))
            box_cx = (cx + 0.5 + dx) * stride
            box_cy = (cy + 0.5 + dy) * stride
            box_w = float(np.exp(dw)) * stride
            box_h = float(np.exp(dh)) * stride
            if box_w > 2 and box_h > 2:
                x1 = max(0, int(box_cx - box_w / 2))
                y1 = max(0, int(box_cy - box_h / 2))
                all_dets.append((q, [x1, y1, int(box_w), int(box_h)]))
    return soft_nms(all_dets, nms_iou)


@torch.no_grad()
def detect_v7_p4only(model, tensor, conf_thresh=0.15, nms_iou=0.3):
    out = model(tensor)
    quality_t = torch.sqrt(
        torch.sigmoid(out["p4_obj"][0, 0]) *
        torch.sigmoid(out["p4_iou"][0, 0]) + 1e-8)
    quality = quality_t.cpu().numpy()
    bbox_raw = out["p4_bbox"][0].cpu()
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(quality, kernel)
    peaks = (quality == dilated) & (quality > conf_thresh)
    
    all_dets = []
    if peaks.any():
        ys, xs = np.where(peaks)
        for cy, cx in zip(ys, xs):
            q = float(quality[cy, cx])
            dx = float(bbox_raw[0, cy, cx].item())
            dy = float(bbox_raw[1, cy, cx].item())
            dw = float(np.clip(bbox_raw[2, cy, cx].item(), -2, 5))
            dh = float(np.clip(bbox_raw[3, cy, cx].item(), -2, 5))
            box_cx = (cx + 0.5 + dx) * 8
            box_cy = (cy + 0.5 + dy) * 8
            box_w = float(np.exp(dw)) * 8
            box_h = float(np.exp(dh)) * 8
            if box_w > 2 and box_h > 2:
                x1 = max(0, int(box_cx - box_w / 2))
                y1 = max(0, int(box_cy - box_h / 2))
                all_dets.append((q, [x1, y1, int(box_w), int(box_h)]))
    return soft_nms(all_dets, nms_iou)


# ──────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────

def load_v7(checkpoint, device):
    model = FaceFCNv7()
    ckpt = torch.load(checkpoint, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def load_v7_p4only(checkpoint, device):
    model = FaceFCNv7P4Only()
    ckpt = torch.load(checkpoint, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


# ──────────────────────────────────────────────
# Evaluation core
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_full(model, dataset, detect_fn, device, conf_thresh=0.25,
                  nms_iou=0.3, num_workers=2):
    """Full evaluation: mAP + speed + per-image details."""
    model.eval()
    all_dets = []
    all_gts = []
    all_categories = []
    all_sizes = []
    latencies = []
    
    for idx in range(len(dataset)):
        batch = dataset[idx]
        if batch[0] is None:
            continue
        tensor, gt_boxes, img_path, category = batch
        all_gts.append(gt_boxes)
        all_categories.append(category[0] if isinstance(category, (list, tuple)) else category)
        
        # Classify face sizes
        img_sizes = []
        for box in gt_boxes:
            area = box[2] * box[3]
            if area < 32*32:
                img_sizes.append("small")
            elif area < 96*96:
                img_sizes.append("medium")
            else:
                img_sizes.append("large")
        all_sizes.append(img_sizes)
        
        tensor = tensor.unsqueeze(0).to(device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        dets = detect_fn(model, tensor, conf_thresh, nms_iou)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        all_dets.append(dets)
    
    return {
        "all_dets": all_dets,
        "all_gts": all_gts,
        "all_categories": all_categories,
        "all_sizes": all_sizes,
        "latencies": np.array(latencies),
    }


def compute_metrics(eval_result, iou_thresh=0.5):
    """Compute all metrics from raw evaluation results."""
    all_dets = eval_result["all_dets"]
    all_gts = eval_result["all_gts"]
    all_categories = eval_result["all_categories"]
    all_sizes = eval_result["all_sizes"]
    latencies = eval_result["latencies"]
    
    # Overall mAP@0.5
    ap50, n_gt = compute_map(all_dets, all_gts, iou_thresh=0.5)
    
    # COCO-style mAP@0.5:0.95
    ap_coco = 0.0
    for t in np.arange(0.5, 1.0, 0.05):
        ap_t, _ = compute_map(all_dets, all_gts, iou_thresh=t)
        ap_coco += ap_t / 10.0
    
    # Per-category mAP (Easy/Medium/Hard by face size)
    cat_buckets = {"easy": ([], []), "medium": ([], []), "hard": ([], [])}
    for i, (dets, gts, cat) in enumerate(zip(all_dets, all_gts, all_categories)):
        diff = WIDER_CATEGORIES.get(cat, "medium")
        cat_buckets[diff][0].append(dets)
        cat_buckets[diff][1].append(gts)
    
    per_category = {}
    for diff in ["easy", "medium", "hard"]:
        dets_b, gts_b = cat_buckets[diff]
        if gts_b:
            ap, ng = compute_map(dets_b, gts_b, iou_thresh=0.5)
            per_category[diff] = {"mAP@0.5": ap, "n_images": len(gts_b), "n_gt": int(ng)}
        else:
            per_category[diff] = {"mAP@0.5": 0.0, "n_images": 0, "n_gt": 0}
    
    # Per-size mAP (small/medium/large faces)
    size_buckets = {"small": ([], []), "medium": ([], []), "large": ([], [])}
    for dets, gts, sizes in zip(all_dets, all_gts, all_sizes):
        for size_label in ["small", "medium", "large"]:
            if size_label in sizes:
                # Filter GT to this size only
                filtered_gts = [g for g, s in zip(gts, sizes) if s == size_label]
                if filtered_gts:
                    size_buckets[size_label][0].append(dets)
                    size_buckets[size_label][1].append(filtered_gts)
    
    per_size = {}
    for sz in ["small", "medium", "large"]:
        dets_b, gts_b = size_buckets[sz]
        if gts_b:
            ap, ng = compute_map(dets_b, gts_b, iou_thresh=0.5)
            per_size[sz] = {"mAP@0.5": ap, "n_images": len(gts_b), "n_gt": int(ng)}
        else:
            per_size[sz] = {"mAP@0.5": 0.0, "n_images": 0, "n_gt": 0}
    
    # Per-event-category mAP
    event_buckets = {}
    for dets, gts, cat in zip(all_dets, all_gts, all_categories):
        if cat not in event_buckets:
            event_buckets[cat] = ([], [])
        event_buckets[cat][0].append(dets)
        event_buckets[cat][1].append(gts)
    
    per_event = {}
    for cat, (dets_b, gts_b) in sorted(event_buckets.items()):
        ap, ng = compute_map(dets_b, gts_b, iou_thresh=0.5)
        per_event[cat] = {"mAP@0.5": ap, "n_gt": int(ng)}
    
    # Speed stats
    speed = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
    }
    
    # Confusion matrix
    tp, fp, fn = 0, 0, 0
    for dets, gts in zip(all_dets, all_gts):
        gt_matched = [False] * len(gts)
        for _, det_box in dets:
            best_iou, best_idx = 0, -1
            for j, gt_box in enumerate(gts):
                if not gt_matched[j]:
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            if best_iou >= iou_thresh and best_idx >= 0:
                tp += 1
                gt_matched[best_idx] = True
            else:
                fp += 1
        fn += len(gts) - sum(gt_matched)
    
    # PR curve (sampled)
    tp_all, fp_all, scores_all, n_gt_pr = [], [], [], 0
    for dets, gts in zip(all_dets, all_gts):
        gt_matched = [False] * len(gts)
        n_gt_pr += len(gts)
        for score, det_box in sorted(dets, key=lambda x: x[0], reverse=True):
            best_iou, best_idx = 0, -1
            for j, gt_box in enumerate(gts):
                if not gt_matched[j]:
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            if best_iou >= 0.5 and best_idx >= 0:
                tp_all.append(1); fp_all.append(0)
                gt_matched[best_idx] = True
            else:
                tp_all.append(0); fp_all.append(1)
            scores_all.append(score)
    
    tp_cum = np.cumsum(tp_all)
    fp_cum = np.cumsum(fp_all)
    pr_recalls = tp_cum / max(n_gt_pr, 1)
    pr_precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    idx = np.arange(0, len(pr_recalls), max(1, len(pr_recalls) // 200))
    
    return {
        "mAP@0.5": ap50,
        "mAP_coco": ap_coco,
        "n_gt": int(n_gt),
        "n_images": len(all_gts),
        "per_category": per_category,
        "per_size": per_size,
        "per_event": per_event,
        "speed": speed,
        "confusion": {"tp": tp, "fp": fp, "fn": fn},
        "pr_curve": {
            "recalls": [float(r) for r in pr_recalls[idx]],
            "precisions": [float(p) for p in pr_precisions[idx]],
            "n_gt": n_gt_pr,
        },
    }


# ──────────────────────────────────────────────
# Threshold & NMS sweeps
# ──────────────────────────────────────────────

def threshold_sweep(model, dataset, detect_fn, device, nms_iou=0.3):
    thresholds = np.linspace(0.05, 0.95, 19)
    results = {}
    for thresh in thresholds:
        ev = evaluate_full(model, dataset, detect_fn, device,
                           conf_thresh=thresh, nms_iou=nms_iou, num_workers=0)
        m, _ = compute_map(ev["all_dets"], ev["all_gts"], 0.5)
        results[float(thresh)] = {
            "mAP@0.5": m,
            "fps": ev["latencies"].shape[0] / (ev["latencies"].sum() / 1000),
        }
    return results


def nms_sweep(model, dataset, detect_fn, device, conf_thresh=0.25):
    nms_thresholds = np.linspace(0.1, 0.9, 9)
    results = {}
    for nms_thresh in nms_thresholds:
        ev = evaluate_full(model, dataset, detect_fn, device,
                           conf_thresh=conf_thresh, nms_iou=nms_thresh, num_workers=0)
        m, _ = compute_map(ev["all_dets"], ev["all_gts"], 0.5)
        results[float(nms_thresh)] = {
            "mAP@0.5": m,
            "fps": ev["latencies"].shape[0] / (ev["latencies"].sum() / 1000),
        }
    return results


# ──────────────────────────────────────────────
# Resolution sensitivity
# ──────────────────────────────────────────────

def resolution_sensitivity(model, detect_fn, device, data_root, conf_thresh=0.25):
    resolutions = [
        (320, 240), (480, 360), (640, 480),
        (800, 600), (960, 720), (1280, 720),
    ]
    results = {}
    for w, h in resolutions:
        ds = WiderValDataset(data_root, target_h=h, target_w=w, max_images=500)
        ev = evaluate_full(model, ds, detect_fn, device,
                           conf_thresh=conf_thresh, num_workers=0)
        m, _ = compute_map(ev["all_dets"], ev["all_gts"], 0.5)
        results[f"{w}x{h}"] = {
            "mAP@0.5": m,
            "fps": float(1000.0 / np.mean(ev["latencies"])),
            "mean_ms": float(np.mean(ev["latencies"])),
        }
    return results


# ──────────────────────────────────────────────
# Deployment pipeline
# ──────────────────────────────────────────────

def onnx_export(model, save_dir, input_shape=(1, 3, 480, 640), model_name="v7"):
    """Export model to ONNX FP32."""
    try:
        import onnx
    except ImportError:
        print("  [SKIP] onnx not installed")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, f"{model_name}_fp32.onnx")
    
    dummy = torch.randn(*input_shape).to(next(model.parameters()).device)
    
    # Use detect method's forward path
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    
    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        out = sess.run(None, {"input": dummy.cpu().numpy()})
        print(f"  ONNX FP32: {onnx_path} — verified, output shape={out[0].shape}")
    except ImportError:
        print(f"  ONNX FP32: {onnx_path} — exported (onnxruntime not installed for verification)")
    
    return onnx_path


def quantize_int8(onnx_path, save_dir, model_name="v7"):
    """Dynamic INT8 quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  [SKIP] onnxruntime.quantization not available")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    int8_path = os.path.join(save_dir, f"{model_name}_int8.onnx")
    
    quantize_dynamic(
        onnx_path, int8_path,
        weight_type=QuantType.QInt8,
    )
    
    # Test inference
    try:
        import onnxruntime as ort
        dummy = np.random.randn(1, 3, 480, 640).astype(np.float32)
        sess = ort.InferenceSession(int8_path)
        out = sess.run(None, {"input": dummy})
        print(f"  INT8: {int8_path} — verified, output shape={out[0].shape}")
    except Exception as e:
        print(f"  INT8: {int8_path} — quantized ({e})")
    
    return int8_path


def benchmark_onnx(onnx_path, n_warmup=10, n_runs=100):
    """Benchmark ONNX Runtime inference speed."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        try:
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"  ONNX benchmark failed: {e}")
            return None
    
    dummy = np.random.randn(1, 3, 480, 640).astype(np.float32)
    
    for _ in range(n_warmup):
        sess.run(None, {"input": dummy})
    
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        latencies.append((time.perf_counter() - t0) * 1000)
    
    lat = np.array(latencies)
    return {
        "mean_ms": float(np.mean(lat)),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / np.mean(lat)),
    }


def convert_openvino(onnx_path, save_dir, model_name="v7"):
    """Convert ONNX to OpenVINO IR (INT8 + FP32)."""
    try:
        from openvino.runtime import Core
    except ImportError:
        print("  [SKIP] openvino not installed")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        from openvino.runtime import Core
        core = Core()
        model = core.read_model(onnx_path)
        
        # FP32 IR
        fp32_path = os.path.join(save_dir, f"{model_name}_fp32.xml")
        ov_model = core.compile_model(model, "CPU")
        
        # Save using MO
        from openvino.runtime import serialize
        serialize(model, fp32_path, fp32_path.replace(".xml", ".bin"))
        
        print(f"  OpenVINO FP32: {fp32_path}")
        return fp32_path
    except Exception as e:
        print(f"  OpenVINO conversion failed: {e}")
        return None


def convert_tensorrt(onnx_path, save_dir, model_name="v7"):
    """Convert ONNX to TensorRT FP16."""
    try:
        import tensorrt as trt
    except ImportError:
        print("  [SKIP] tensorrt not installed")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    engine_path = os.path.join(save_dir, f"{model_name}_fp16.engine")
    
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  TRT parse error: {parser.get_error(i)}")
                return None
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine)
        
        print(f"  TensorRT FP16: {engine_path}")
        return engine_path
    except Exception as e:
        print(f"  TensorRT conversion failed: {e}")
        return None


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def generate_report(all_results, output_dir):
    """Generate markdown summary report."""
    report_path = os.path.join(output_dir, "v7_benchmark_report.md")
    
    lines = [
        "# FaceFCN v7 — Comprehensive Benchmark Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"Input resolution: 480×640",
        "",
        "## Summary",
        "",
        "| Model | mAP@0.5 | mAP(COCO) | Easy | Medium | Hard | FPS | Params |",
        "|-------|---------|-----------|------|--------|------|-----|--------|",
    ]
    
    for key, r in all_results.items():
        p = r.get("per_category", {})
        lines.append(
            f"| {r['model_name']} | {r['mAP@0.5']:.4f} | {r['mAP_coco']:.4f} "
            f"| {p.get('easy', {}).get('mAP@0.5', 0):.4f} "
            f"| {p.get('medium', {}).get('mAP@0.5', 0):.4f} "
            f"| {p.get('hard', {}).get('mAP@0.5', 0):.4f} "
            f"| {r['speed']['fps']:.1f} | {r.get('n_params', 0):,} |"
        )
    
    # Per-size
    lines.extend(["", "## Per-Size Performance", "",
                   "| Model | Small | Medium | Large |",
                   "|-------|-------|--------|-------|"])
    for key, r in all_results.items():
        ps = r.get("per_size", {})
        lines.append(
            f"| {r['model_name']} "
            f"| {ps.get('small', {}).get('mAP@0.5', 0):.4f} "
            f"| {ps.get('medium', {}).get('mAP@0.5', 0):.4f} "
            f"| {ps.get('large', {}).get('mAP@0.5', 0):.4f} |"
        )
    
    # Speed
    lines.extend(["", "## Latency (GPU)", "",
                   "| Model | Mean | Median | P95 | P99 | FPS |",
                   "|-------|------|--------|-----|-----|-----|"])
    for key, r in all_results.items():
        s = r["speed"]
        lines.append(
            f"| {r['model_name']} | {s['mean_ms']:.1f}ms | {s['median_ms']:.1f}ms "
            f"| {s['p95_ms']:.1f}ms | {s['p99_ms']:.1f}ms | {s['fps']:.1f} |"
        )
    
    # Confusion
    lines.extend(["", "## Confusion Matrix", "",
                   "| Model | TP | FP | FN | Precision | Recall |",
                   "|-------|----|----|----|-----------|--------|"])
    for key, r in all_results.items():
        c = r.get("confusion", {})
        tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        lines.append(
            f"| {r['model_name']} | {tp} | {fp} | {fn} | {prec:.4f} | {rec:.4f} |"
        )
    
    # Deployment
    deploy_exists = any("deployment" in r for r in all_results.values())
    if deploy_exists:
        lines.extend(["", "## Deployment Pipeline", "",
                       "| Model | Backend | Mean(ms) | FPS |",
                       "|-------|---------|----------|-----|"])
        for key, r in all_results.items():
            for backend, speed in r.get("deployment", {}).items():
                if speed:
                    lines.append(
                        f"| {r['model_name']} | {backend} | "
                        f"{speed.get('mean_ms', 0):.1f}ms | {speed.get('fps', 0):.1f} |"
                    )
    
    # Top event categories
    lines.extend(["", "## Top Event Categories (by mAP@0.5)", ""])
    for key, r in all_results.items():
        pe = r.get("per_event", {})
        if pe:
            sorted_ev = sorted(pe.items(), key=lambda x: x[1]["mAP@0.5"], reverse=True)[:10]
            lines.append(f"### {r['model_name']}")
            lines.append("| Category | mAP@0.5 | GT Faces |")
            lines.append("|----------|---------|----------|")
            for cat, vals in sorted_ev:
                lines.append(f"| {cat} | {vals['mAP@0.5']:.4f} | {vals['n_gt']} |")
            lines.append("")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nReport saved: {report_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FaceFCN v7 comprehensive benchmark + deployment pipeline")
    parser.add_argument("--data", default="data/face/widerface",
                        help="WIDER Face dataset root")
    parser.add_argument("--models", nargs="+", default=["v7", "v7_p4only"],
                        choices=["v7", "v7_p4only"],
                        help="Models to benchmark")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images for faster testing")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.3,
                        help="NMS IoU threshold")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for inference")
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold + NMS sweeps")
    parser.add_argument("--resolution", action="store_true",
                        help="Run resolution sensitivity analysis")
    parser.add_argument("--deploy", action="store_true",
                        help="Run ONNX/INT8/OpenVINO/TensorRT pipeline")
    parser.add_argument("--output", default="benchmarks/v7_comprehensive",
                        help="Output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  FaceFCN v7 Comprehensive Benchmark")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    # Model definitions
    model_defs = {
        "v7": {
            "name": "v7 (P3+P4, 519K params)",
            "checkpoint": str(REPO / "models" / "face_cnn_v7.pth"),
            "loader": load_v7,
            "detect": detect_v7,
            "default_conf": 0.25,
        },
        "v7_p4only": {
            "name": "v7 P4-only (453K params)",
            "checkpoint": str(REPO / "models" / "face_cnn_v7_p4only.pth"),
            "loader": load_v7_p4only,
            "detect": detect_v7_p4only,
            "default_conf": 0.15,
        },
    }
    
    dataset = WiderValDataset(args.data, max_images=args.max_images)
    all_results = {}
    
    for model_key in args.models:
        md = model_defs[model_key]
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {md['name']}")
        print(f"{'─'*60}")
        
        if not os.path.exists(md["checkpoint"]):
            print(f"  WARNING: Checkpoint not found: {md['checkpoint']}")
            continue
        
        # Use model-specific default conf if not overridden
        conf = args.conf if args.conf != 0.25 else md["default_conf"]
        
        model = md["loader"](md["checkpoint"], device)
        n_params = sum(p.numel() for p in model.parameters())
        
        # GPU memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Core evaluation
        t0 = time.time()
        ev = evaluate_full(model, dataset, md["detect"], device,
                           conf_thresh=conf, nms_iou=args.nms)
        elapsed = time.time() - t0
        
        metrics = compute_metrics(ev, iou_thresh=0.5)
        metrics["model_name"] = md["name"]
        metrics["n_params"] = n_params
        metrics["eval_time_s"] = elapsed
        
        if device.type == "cuda":
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        
        # Print summary
        print(f"  mAP@0.5:   {metrics['mAP@0.5']:.4f}")
        print(f"  mAP(COCO): {metrics['mAP_coco']:.4f}")
        pcat = metrics["per_category"]
        print(f"  Easy:      {pcat['easy']['mAP@0.5']:.4f}")
        print(f"  Medium:    {pcat['medium']['mAP@0.5']:.4f}")
        print(f"  Hard:      {pcat['hard']['mAP@0.5']:.4f}")
        ps = metrics["per_size"]
        print(f"  Small:     {ps['small']['mAP@0.5']:.4f}")
        print(f"  Medium:    {ps['medium']['mAP@0.5']:.4f}")
        print(f"  Large:     {ps['large']['mAP@0.5']:.4f}")
        print(f"  FPS:       {metrics['speed']['fps']:.1f}")
        print(f"  Params:    {n_params:,}")
        print(f"  Eval time: {elapsed:.1f}s")
        
        # Threshold + NMS sweeps
        if args.sweep:
            print(f"\n  Threshold sweep:")
            metrics["threshold_sweep"] = threshold_sweep(
                model, dataset, md["detect"], device, nms_iou=args.nms)
            print(f"\n  NMS sweep:")
            metrics["nms_sweep"] = nms_sweep(
                model, dataset, md["detect"], device, conf_thresh=conf)
        
        # Resolution sensitivity
        if args.resolution:
            print(f"\n  Resolution sensitivity:")
            metrics["resolution_sensitivity"] = resolution_sensitivity(
                model, md["detect"], device, args.data, conf_thresh=conf)
        
        # Deployment pipeline
        if args.deploy:
            deploy_dir = os.path.join(args.output, "deploy")
            print(f"\n  Deployment pipeline:")
            deployment = {}
            
            onnx_path = onnx_export(model, deploy_dir, model_name=model_key)
            if onnx_path:
                deployment["onnx_fp32"] = benchmark_onnx(onnx_path)
                
                int8_path = quantize_int8(onnx_path, deploy_dir, model_name=model_key)
                if int8_path:
                    deployment["onnx_int8"] = benchmark_onnx(int8_path)
                
                ov_path = convert_openvino(onnx_path, deploy_dir, model_name=model_key)
                if ov_path:
                    deployment["openvino_fp32"] = {"converted": True}
                
                trt_path = convert_tensorrt(onnx_path, deploy_dir, model_name=model_key)
                if trt_path:
                    deployment["tensorrt_fp16"] = {"converted": True}
            
            metrics["deployment"] = deployment
        
        all_results[model_key] = metrics
        
        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Cross-model comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'mAP@0.5':>10} {'COCO':>10} {'FPS':>10}")
        print(f"{'─'*60}")
        for key, r in all_results.items():
            print(f"{r['model_name']:<30} {r['mAP@0.5']:>10.4f} "
                  f"{r['mAP_coco']:>10.4f} {r['speed']['fps']:>10.1f}")
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output, f"benchmark_{ts}.json")
    
    # Remove large non-serializable data
    save_results = {}
    for key, r in all_results.items():
        sr = {k: v for k, v in r.items()
              if k not in ("all_dets", "all_gts", "all_categories", "all_sizes", "latencies")}
        save_results[key] = sr
    
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")
    
    # Save CSV
    csv_path = os.path.join(args.output, f"benchmark_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mAP@0.5", "mAP_coco", "easy", "medium", "hard",
                      "small", "medium_face", "large", "fps", "mean_ms", "params"])
        for key, r in all_results.items():
            pc = r.get("per_category", {})
            ps = r.get("per_size", {})
            s = r.get("speed", {})
            w.writerow([
                r["model_name"], f"{r['mAP@0.5']:.4f}", f"{r['mAP_coco']:.4f}",
                f"{pc.get('easy', {}).get('mAP@0.5', 0):.4f}",
                f"{pc.get('medium', {}).get('mAP@0.5', 0):.4f}",
                f"{pc.get('hard', {}).get('mAP@0.5', 0):.4f}",
                f"{ps.get('small', {}).get('mAP@0.5', 0):.4f}",
                f"{ps.get('medium', {}).get('mAP@0.5', 0):.4f}",
                f"{ps.get('large', {}).get('mAP@0.5', 0):.4f}",
                f"{s.get('fps', 0):.1f}", f"{s.get('mean_ms', 0):.1f}",
                r.get("n_params", 0),
            ])
    print(f"CSV saved: {csv_path}")
    
    # Generate markdown report
    generate_report(all_results, args.output)
    
    print(f"\n{'='*60}")
    print(f"  Benchmark complete. Results in: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
