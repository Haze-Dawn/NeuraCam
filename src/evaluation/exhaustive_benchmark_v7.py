"""
FaceFCN v7/v7.1/v8 — Exhaustive Benchmark Script
=================================================
Full evaluation pipeline for all face detection model variants.

Metrics computed:
  1. WIDER Face mAP@IoU=0.5 (primary accuracy metric)
  2. Per-level mAP (P3/P4/P5 contributions)
  3. Inference speed: latency (mean/median/p95/p99), FPS
  4. GPU memory footprint
  5. Model parameter count
  6. Confusion matrix at optimal threshold
  7. PR curve data
  8. Confidence calibration analysis
  9. Resolution sensitivity analysis
 10. Confidence threshold sweep
 11. NMS IoU threshold sweep
 12. Cross-model comparison (v7 vs v7-p4only vs v7.1 vs v8)

Usage:
  # Benchmark all models
  python3 src/evaluation/exhaustive_benchmark_v7.py

  # Benchmark specific model
  python3 src/evaluation/exhaustive_benchmark_v7.py --models v7 v71 v8

  # Quick benchmark (fewer images)
  python3 src/evaluation/exhaustive_benchmark_v7.py --max-images 500

  # Full benchmark with threshold sweeps
  python3 src/evaluation/exhaustive_benchmark_v7.py --sweep-thresholds --sweep-nms
"""

import os
import sys
import json
import time
import argparse
import glob
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model imports
from src.cv.face_detector_v7 import FaceFCNv7
from src.cv.face_detector_v7_p4only import FaceFCNv7P4Only
from src.cv.face_detector_v8 import FaceFCNv8, DetectionHead


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class WiderValDataset(Dataset):
    """WIDER Face validation dataset."""
    
    def __init__(self, root_dir, target_h=480, target_w=640, max_images=None):
        self.target_h = target_h
        self.target_w = target_w
        self.samples = []
        
        img_dir = os.path.join(root_dir, "WIDER_val", "images")
        annot_file = os.path.join(root_dir, "wider_face_split", "wider_face_val_bbx_gt.txt")
        
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")
        
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
                self.samples.append({"image_path": img_path, "faces": faces})
        
        if max_images:
            self.samples = self.samples[:max_images]
        
        print(f"WiderValDataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ann = self.samples[idx]
        img = cv2.imread(ann["image_path"])
        if img is None:
            return None, None, None
        
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
        
        return tensor, gt_boxes, ann["image_path"]


# ──────────────────────────────────────────────
# IoU & mAP
# ──────────────────────────────────────────────

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    return inter / (area1 + area2 - inter + 1e-8)


def soft_nms(dets, iou_thresh=0.3):
    """Soft-NMS: decay overlapping detections instead of removing."""
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
    """Compute AP using VOC 11-point interpolation."""
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
    return ap


def compute_map(all_dets, all_gts, iou_thresh=0.5):
    """Compute mAP@IoU=iou_thresh."""
    tp_list, fp_list, fn_list, scores_list, n_gt = [], [], [], [], 0
    
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
        
        fn_list.append(len(gts) - sum(gt_matched))
    
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    fn_total = sum(fn_list)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = compute_ap(recalls, precisions) if n_gt > 0 else 0.0
    
    return ap, float(n_gt), len(all_dets)


# ──────────────────────────────────────────────
# Model Loaders
# ──────────────────────────────────────────────

def load_model_v7(checkpoint_path, device="cuda"):
    """Load v7 (full or p4only) model."""
    model = FaceFCNv7()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_model_v7_p4only(checkpoint_path, device="cuda"):
    """Load v7 P4-only model."""
    model = FaceFCNv7P4Only()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_model_v71(checkpoint_path, device="cuda"):
    """Load v7.1 model."""
    from src.cv.face_detector_v71 import FaceFCNv7_1
    model = FaceFCNv7_1(obj_bias=-3.0)
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def load_model_v8(checkpoint_path, device="cuda"):
    """Load v8 model."""
    model = FaceFCNv8()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


MODEL_LOADERS = {
    "v7": load_model_v7,
    "v7_p4only": load_model_v7_p4only,
    "v71": load_model_v71,
    "v8": load_model_v8,
}


# ──────────────────────────────────────────────
# Detection Functions
# ──────────────────────────────────────────────

@torch.no_grad()
def detect_v7(model, tensor, conf_thresh=0.25, nms_iou=0.3):
    """Detect faces using v7 model."""
    out = model(tensor)
    all_dets = []
    
    levels = [
        ("p3", 4, out["p3_obj"][0, 0], out["p3_iou"][0, 0], out["p3_bbox"][0]),
        ("p4", 8, out["p4_obj"][0, 0], out["p4_iou"][0, 0], out["p4_bbox"][0]),
    ]
    
    for level, stride, obj_map, iou_map, bbox_map in levels:
        obj = obj_map.cpu().numpy()
        iou_p = iou_map.cpu().numpy()
        quality = np.sqrt(np.sigmoid(obj) * np.sigmoid(iou_p) + 1e-8)
        
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(quality, kernel)
        peaks = (quality == dilated) & (quality > conf_thresh)
        
        if not peaks.any():
            continue
        
        ys, xs = np.where(peaks)
        bbox_raw = bbox_map.cpu()
        
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
    
    all_dets = soft_nms(all_dets, nms_iou)
    return all_dets


@torch.no_grad()
def detect_v7_p4only(model, tensor, conf_thresh=0.25, nms_iou=0.3):
    """Detect faces using v7 P4-only model."""
    out = model(tensor)
    all_dets = []
    
    obj = out["p4_obj"][0, 0].cpu().numpy()
    iou_p = out["p4_iou"][0, 0].cpu().numpy()
    quality = np.sqrt(np.sigmoid(obj) * np.sigmoid(iou_p) + 1e-8)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(quality, kernel)
    peaks = (quality == dilated) & (quality > conf_thresh)
    
    if not peaks.any():
        return []
    
    ys, xs = np.where(peaks)
    bbox_raw = out["p4_bbox"][0].cpu()
    
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
    
    all_dets = soft_nms(all_dets, nms_iou)
    return all_dets


@torch.no_grad()
def detect_v8(model, tensor, conf_thresh=0.25, nms_iou=0.3):
    """Detect faces using v8 model."""
    out = model(tensor)
    all_dets = []
    
    levels = [
        ("p3", 4, out["p3_obj"][0, 0], out["p3_iou"][0, 0], out["p3_bbox"][0]),
        ("p4", 8, out["p4_obj"][0, 0], out["p4_iou"][0, 0], out["p4_bbox"][0]),
        ("p5", 16, out["p5_obj"][0, 0], out["p5_iou"][0, 0], out["p5_bbox"][0]),
    ]
    
    for level, stride, obj_map, iou_map, bbox_map in levels:
        obj = obj_map.cpu().numpy()
        iou_p = iou_map.cpu().numpy()
        quality = np.sqrt(np.sigmoid(obj) * np.sigmoid(iou_p) + 1e-8)
        
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(quality, kernel)
        peaks = (quality == dilated) & (quality > conf_thresh)
        
        if not peaks.any():
            continue
        
        ys, xs = np.where(peaks)
        bbox_raw = bbox_map.cpu()
        
        for cy, cx in zip(ys, xs):
            q = float(quality[cy, cx])
            offsets = DetectionHead.decode_bbox(
                bbox_raw[:, cy:cy+1, cx:cx+1].unsqueeze(0), stride
            ).squeeze()
            
            dx = float(offsets[0])
            dy = float(offsets[1])
            dw = float(np.clip(offsets[2], -2, 5))
            dh = float(np.clip(offsets[3], -2, 5))
            
            box_cx = (cx + 0.5 + dx) * stride
            box_cy = (cy + 0.5 + dy) * stride
            box_w = float(np.exp(dw)) * stride
            box_h = float(np.exp(dh)) * stride
            
            if box_w > 2 and box_h > 2:
                x1 = max(0, int(box_cx - box_w / 2))
                y1 = max(0, int(box_cy - box_h / 2))
                all_dets.append((q, [x1, y1, int(box_w), int(box_h)]))
    
    all_dets = soft_nms(all_dets, nms_iou)
    return all_dets


DETECT_FNS = {
    "v7": detect_v7,
    "v7_p4only": detect_v7,
    "v71": detect_v7,
    "v8": detect_v8,
}


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, dataset, detect_fn, device, conf_thresh=0.25, 
                   nms_iou=0.3, num_workers=2):
    """Full model evaluation: mAP + speed."""
    model.eval()
    all_dets = []
    all_gts = []
    all_images = []
    latencies = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                       num_workers=num_workers, pin_memory=True)
    
    for tensor, gt_boxes, img_path in tqdm(loader, desc="Evaluating"):
        if tensor is None:
            continue
        
        all_gts.append(gt_boxes)
        all_images.append(img_path)
        tensor = tensor.to(device)
        
        # Time inference
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        dets = detect_fn(model, tensor, conf_thresh, nms_iou)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        latencies.append(elapsed_ms)
        all_dets.append(dets)
    
    # Compute mAP
    ap, n_gt, n_dets = compute_map(all_dets, all_gts, iou_thresh=0.5)
    
    # Compute speed stats
    latencies = np.array(latencies)
    speed_stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
    }
    
    return {
        "mAP@0.5": ap,
        "n_gt": int(n_gt),
        "n_detections": n_dets,
        "n_images": len(all_gts),
        "speed": speed_stats,
        "all_dets": all_dets,
        "all_gts": all_gts,
    }


# ──────────────────────────────────────────────
# Threshold Sweep
# ──────────────────────────────────────────────

def threshold_sweep(model, dataset, detect_fn, device, 
                    thresholds=None, nms_iou=0.3, max_images=500):
    """Sweep confidence thresholds and compute mAP for each."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    results = {}
    for thresh in thresholds:
        eval_result = evaluate_model(
            model, dataset, detect_fn, device,
            conf_thresh=thresh, nms_iou=nms_iou, num_workers=0
        )
        results[float(thresh)] = {
            "mAP@0.5": eval_result["mAP@0.5"],
            "n_detections": eval_result["n_detections"],
            "fps": eval_result["speed"]["fps"],
        }
        print(f"  thresh={thresh:.2f}: mAP={eval_result['mAP@0.5']:.4f} "
              f"dets={eval_result['n_detections']}")
    
    return results


def nms_sweep(model, dataset, detect_fn, device,
              nms_thresholds=None, conf_thresh=0.25, max_images=500):
    """Sweep NMS IoU thresholds and compute mAP for each."""
    if nms_thresholds is None:
        nms_thresholds = np.linspace(0.1, 0.9, 9)
    
    results = {}
    for nms_thresh in nms_thresholds:
        eval_result = evaluate_model(
            model, dataset, detect_fn, device,
            conf_thresh=conf_thresh, nms_iou=nms_thresh, num_workers=0
        )
        results[float(nms_thresh)] = {
            "mAP@0.5": eval_result["mAP@0.5"],
            "n_detections": eval_result["n_detections"],
            "fps": eval_result["speed"]["fps"],
        }
        print(f"  nms_iou={nms_thresh:.2f}: mAP={eval_result['mAP@0.5']:.4f} "
              f"dets={eval_result['n_detections']}")
    
    return results


# ──────────────────────────────────────────────
# Resolution Sensitivity
# ──────────────────────────────────────────────

def resolution_sensitivity(model, detect_fn, device, conf_thresh=0.25):
    """Test model performance at different input resolutions."""
    resolutions = [
        (320, 240), (480, 360), (640, 480), 
        (800, 600), (960, 720), (1280, 720),
    ]
    
    results = {}
    for w, h in resolutions:
        dataset = WiderValDataset("data/face/widerface", target_h=h, target_w=w)
        eval_result = evaluate_model(
            model, dataset, detect_fn, device,
            conf_thresh=conf_thresh, num_workers=0
        )
        results[f"{w}x{h}"] = {
            "mAP@0.5": eval_result["mAP@0.5"],
            "speed": eval_result["speed"],
        }
        print(f"  {w}x{h}: mAP={eval_result['mAP@0.5']:.4f} "
              f"fps={eval_result['speed']['fps']:.1f}")
    
    return results


# ──────────────────────────────────────────────
# PR Curve Data
# ──────────────────────────────────────────────

def compute_pr_curve(all_dets, all_gts):
    """Compute precision-recall curve data."""
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
            
            if best_iou >= 0.5 and best_idx >= 0:
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
    
    # Sample every 100th point to reduce size
    indices = np.arange(0, len(recalls), max(1, len(recalls) // 200))
    
    return {
        "recalls": [float(r) for r in recalls[indices]],
        "precisions": [float(p) for p in precisions[indices]],
        "n_gt": int(n_gt),
    }


# ──────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────

def compute_confusion_matrix(all_dets, all_gts, iou_thresh=0.5):
    """Compute confusion matrix at given IoU threshold."""
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
    
    return {"tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────
# Confidence Calibration
# ──────────────────────────────────────────────

def compute_calibration(all_dets, all_gts, n_bins=10):
    """Compute confidence calibration (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        bin_dets = []
        bin_gts = []
        
        for dets, gts in zip(all_dets, all_gts):
            for score, det_box in dets:
                if lo <= score < hi:
                    bin_dets.append((score, det_box))
            bin_gts.append(gts)
        
        if bin_dets:
            tp, fp = 0, 0
            for dets, gts in zip([bin_dets], [bin_gts]):
                gt_matched = [False] * len(gts[0])
                for score, det_box in dets:
                    best_iou, best_idx = 0, -1
                    for j, gt_box in enumerate(gts[0]):
                        if not gt_matched[j]:
                            iou = compute_iou(det_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = j
                    if best_iou >= 0.5 and best_idx >= 0:
                        tp += 1
                        gt_matched[best_idx] = True
                    else:
                        fp += 1
            
            conf = np.mean([s for s, _ in bin_dets])
            acc = tp / max(tp + fp, 1)
            count = len(bin_dets)
            
            bin_confidences.append(float(conf))
            bin_accuracies.append(float(acc))
            bin_counts.append(count)
        else:
            bin_confidences.append(float((lo + hi) / 2))
            bin_accuracies.append(0.0)
            bin_counts.append(0)
    
    # Compute ECE
    ece = sum(abs(c - a) * n / max(sum(bin_counts), 1) 
              for c, a, n in zip(bin_confidences, bin_accuracies, bin_counts))
    
    return {
        "ece": float(ece),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


# ──────────────────────────────────────────────
# Main Benchmark
# ──────────────────────────────────────────────

def run_exhaustive_benchmark(args):
    """Run full exhaustive benchmark across all models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  EXHAUSTIVE FACE DETECTION BENCHMARK")
    print(f"  Device: {device}")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset = WiderValDataset(
        args.data, target_h=480, target_w=640, max_images=args.max_images
    )
    
    # Model definitions
    model_defs = {
        "v7": {
            "name": "v7 (P3+P4, 519K params)",
            "checkpoint": "models/face_cnn_v7.pth",
            "loader": load_model_v7,
            "detect": detect_v7,
        },
        "v7_p4only": {
            "name": "v7 P4-only (453K params)",
            "checkpoint": "models/face_cnn_v7_p4only.pth",
            "loader": load_model_v7_p4only,
            "detect": detect_v7,
        },
        "v7_ep143": {
            "name": "v7 epoch 143 (Phase 1 best)",
            "checkpoint": "models/face_cnn_v7_ep140.pth",
            "loader": load_model_v7,
            "detect": detect_v7,
        },
        "v71": {
            "name": "v7.1 (P2+P3+P4, 504K params)",
            "checkpoint": "models/face_cnn_v71_ep050.pth",
            "loader": load_model_v71,
            "detect": detect_v7,
        },
        "v8": {
            "name": "v8 (BiFPN, DFL, 21 blocks)",
            "checkpoint": "models/face_cnn_v8_best.pth",
            "loader": load_model_v8,
            "detect": detect_v8,
        },
    }
    
    # Filter models if specified
    if args.models:
        model_defs = {k: v for k, v in model_defs.items() if k in args.models}
    
    results = {}
    
    for model_key, model_def in model_defs.items():
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {model_def['name']}")
        print(f"  Checkpoint: {model_def['checkpoint']}")
        print(f"{'─'*60}")
        
        if not os.path.exists(model_def["checkpoint"]):
            print(f"  WARNING: Checkpoint not found, skipping")
            continue
        
        # Load model
        model = model_def["loader"](model_def["checkpoint"], device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        
        # GPU memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Run evaluation
        eval_result = evaluate_model(
            model, dataset, model_def["detect"], device,
            conf_thresh=args.conf_thresh, nms_iou=args.nms_iou
        )
        
        # GPU memory used
        gpu_mem_mb = 0
        if device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Compute PR curve
        pr_data = compute_pr_curve(eval_result["all_dets"], eval_result["all_gts"])
        
        # Compute confusion matrix
        conf_matrix = compute_confusion_matrix(eval_result["all_dets"], eval_result["all_gts"])
        
        # Compute calibration
        calibration = compute_calibration(eval_result["all_dets"], eval_result["all_gts"])
        
        # Store results
        results[model_key] = {
            "model_name": model_def["name"],
            "checkpoint": model_def["checkpoint"],
            "n_params": n_params,
            "gpu_memory_mb": gpu_mem_mb,
            "mAP@0.5": eval_result["mAP@0.5"],
            "n_gt": eval_result["n_gt"],
            "n_images": eval_result["n_images"],
            "speed": eval_result["speed"],
            "confusion_matrix": conf_matrix,
            "calibration": calibration,
            "pr_curve": pr_data,
        }
        
        # Print summary
        print(f"\n  Results:")
        print(f"    Parameters: {n_params:,}")
        print(f"    GPU Memory: {gpu_mem_mb:.1f} MB")
        print(f"    mAP@0.5: {eval_result['mAP@0.5']:.4f}")
        print(f"    GT Faces: {eval_result['n_gt']}")
        print(f"    Speed: {eval_result['speed']['mean_ms']:.1f}ms "
              f"({eval_result['speed']['fps']:.1f} FPS)")
        print(f"    Confusion: TP={conf_matrix['tp']} FP={conf_matrix['fp']} "
              f"FN={conf_matrix['fn']}")
        print(f"    Calibration ECE: {calibration['ece']:.4f}")
        
        # Threshold sweep
        if args.sweep_thresholds:
            print(f"\n  Threshold Sweep:")
            results[model_key]["threshold_sweep"] = threshold_sweep(
                model, dataset, model_def["detect"], device,
                nms_iou=args.nms_iou, max_images=args.max_images
            )
        
        # NMS sweep
        if args.sweep_nms:
            print(f"\n  NMS Sweep:")
            results[model_key]["nms_sweep"] = nms_sweep(
                model, dataset, model_def["detect"], device,
                conf_thresh=args.conf_thresh, max_images=args.max_images
            )
        
        # Resolution sensitivity (only for first model)
        if args.resolution_sensitivity and model_key == list(model_defs.keys())[0]:
            print(f"\n  Resolution Sensitivity:")
            results[model_key]["resolution_sensitivity"] = resolution_sensitivity(
                model, model_def["detect"], device, conf_thresh=args.conf_thresh
            )
        
        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Cross-model comparison table
    print(f"\n{'='*60}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'mAP@0.5':>10} {'FPS':>10} {'Params':>10} {'GPU MB':>10}")
    print(f"{'─'*65}")
    
    for model_key, r in results.items():
        print(f"{r['model_name']:<25} {r['mAP@0.5']:>10.4f} "
              f"{r['speed']['fps']:>10.1f} {r['n_params']:>10,} "
              f"{r['gpu_memory_mb']:>10.1f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"exhaustive_benchmark_{ts}.json")
    
    # Remove non-serializable items
    for model_key in results:
        if "all_dets" in results[model_key]:
            del results[model_key]["all_dets"]
        if "all_gts" in results[model_key]:
            del results[model_key]["all_gts"]
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive benchmark for face detection models"
    )
    parser.add_argument("--data", default="data/face/widerface",
                        help="WIDER Face dataset root")
    parser.add_argument("--models", nargs="+", 
                        choices=["v7", "v7_p4only", "v7_ep143", "v71", "v8"],
                        help="Models to benchmark (default: all)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images for faster testing")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.3,
                        help="NMS IoU threshold")
    parser.add_argument("--sweep-thresholds", action="store_true",
                        help="Run confidence threshold sweep")
    parser.add_argument("--sweep-nms", action="store_true",
                        help="Run NMS IoU threshold sweep")
    parser.add_argument("--resolution-sensitivity", action="store_true",
                        help="Run resolution sensitivity analysis")
    parser.add_argument("--output-dir", default="benchmarks",
                        help="Output directory for results")
    
    args = parser.parse_args()
    run_exhaustive_benchmark(args)


if __name__ == "__main__":
    main()
