"""
FaceCNN v7.1 — Dual-Level Training Pipeline (P2+P4, Mosaic, Cosine+SWA)
==============================================================================
Phase 1 — WIDER-only (200 epochs, ~14h):
  Dual-level: P4 (stride 4, 5-layer DW head) + P2 (stride 2, 2-layer head)
  Loss: VarifocalLoss(obj) + 0.5xMSE(iou) + 5.0xEIoU(bbox)
  Label: ATSS with gaussian fallback (min 8 positives per face)
  Mosaic augmentation (50% prob, 4-image composite)
  Multi-scale training (384-800px random resize + crop)
  Copy-paste augmentation (+5 faces per image from batch)
  EMA inference weights (decay=0.999, BN buffers synced)
  Cosine annealing + SWA (swa_start=150)
  Stage-level gradient checkpointing + FP32
  Logit clamp [-10, 10], NaN guard with counter
  Hard-aware sample redistribution (after epoch 50)

Target: Beat SCRFD-0.5GF on WIDER Face Ease/Medium/Hard.
"""

import os, sys, time, csv, math, argparse, json, warnings, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv.face_detector_v71 import FaceFCNv7_1

warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")


# ============================================================================
# VarifocalLoss (SCRFD-style: sum over all cells / num_pos)
# ============================================================================

class VarifocalLoss(nn.Module):
    """VFL with sum/num_pos reduction — matches SCRFD-0.5GF exactly.
    sum(VFL) / num_pos amplifies positive gradient by ~400:1 class ratio.
    Without this: mean() over 19200 cells dilutes positive gradient to zero."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred_logits, targets, pos_mask):
        eps = 1e-3
        pred = torch.sigmoid(pred_logits).clamp(eps, 1 - eps)
        fg_weight = targets * (1 - pred) ** self.gamma
        bg_weight = (1 - targets) * pred ** self.gamma
        loss = -(fg_weight * torch.log(pred) + bg_weight * torch.log(1 - pred))
        n_pos = pos_mask.float().sum().clamp(min=1)
        return loss.sum() / n_pos


# ============================================================================
# EIoU Loss (on decoded boxes)
# ============================================================================

class EIoULoss(nn.Module):
    def forward(self, pred_bbox, target_bbox, pos_mask, stride):
        pm = pos_mask > 0.5
        if pm.sum() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)
        pred_bbox = pred_bbox.float()
        target_bbox = target_bbox.float()
        B = pred_bbox.size(0); device = pred_bbox.device
        gh, gw = pred_bbox.size(-2), pred_bbox.size(-1)
        ys, xs = torch.meshgrid(torch.arange(gh, device=device, dtype=torch.float32),
                                torch.arange(gw, device=device, dtype=torch.float32),
                                indexing="ij")
        xs = xs.unsqueeze(0).expand(B, -1, -1); ys = ys.unsqueeze(0).expand(B, -1, -1)
        pd = pred_bbox[:, 0]; qd = pred_bbox[:, 1]
        pw_ = pred_bbox[:, 2].clamp(max=5.0); ph_ = pred_bbox[:, 3].clamp(max=5.0)
        p_cx = (xs + 0.5 + pd) * stride; p_cy = (ys + 0.5 + qd) * stride
        p_w = torch.exp(pw_) * stride; p_h = torch.exp(ph_) * stride
        td = target_bbox[:, 0]; ud = target_bbox[:, 1]
        tw_ = target_bbox[:, 2]; th_ = target_bbox[:, 3]
        t_cx = (xs + 0.5 + td) * stride; t_cy = (ys + 0.5 + ud) * stride
        t_w = torch.exp(tw_) * stride; t_h = torch.exp(th_) * stride
        p_x1 = p_cx - p_w/2; p_y1 = p_cy - p_h/2
        p_x2 = p_cx + p_w/2; p_y2 = p_cy + p_h/2
        t_x1 = t_cx - t_w/2; t_y1 = t_cy - t_h/2
        t_x2 = t_cx + t_w/2; t_y2 = t_cy + t_h/2
        inter_x1 = torch.max(p_x1, t_x1); inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2); inter_y2 = torch.min(p_y2, t_y2)
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = p_w * p_h + t_w * t_h - inter + 1e-8; iou = inter / union
        c_x1 = torch.min(p_x1, t_x1); c_y1 = torch.min(p_y1, t_y1)
        c_x2 = torch.max(p_x2, t_x2); c_y2 = torch.max(p_y2, t_y2)
        c_diag = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2) + 1e-8
        center_dist = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)
        c_w = (c_x2 - c_x1).clamp(min=1); c_h = (c_y2 - c_y1).clamp(min=1)
        w_dist = (p_w - t_w).pow(2) / (c_w.pow(2) + 1e-8)
        h_dist = (p_h - t_h).pow(2) / (c_h.pow(2) + 1e-8)
        eiou = iou - center_dist / c_diag - w_dist - h_dist
        eiou = torch.nan_to_num(eiou, nan=0.0, posinf=0.0, neginf=0.0)
        loss_per = (1 - eiou) * pm.float().squeeze(1)
        n_pos = pm.sum().clamp(min=1)
        return loss_per.sum() / n_pos


# ============================================================================
# Combined V7.1 Loss (Dual-level: P4 + P2)
# ============================================================================

class V7_1Loss(nn.Module):
    def __init__(self, varifocal_gamma=2.0, p2_weight=0.5):
        super().__init__()
        self.criterion_vfl = VarifocalLoss(gamma=varifocal_gamma)
        self.criterion_eiou = EIoULoss()
        self.p2_weight = p2_weight

    def forward(self, out_dict, targets, strides={"p4": 4, "p2": 2}):
        device = next(iter(out_dict.values())).device
        total_loss = torch.tensor(0.0, device=device)
        comp = {}
        for level, stride in strides.items():
            prefix = "" if level == "p4" else "p2_"
            o = torch.clamp(out_dict[f"{prefix}obj"], -10, 10)
            i = out_dict[f"{prefix}iou"]
            t = targets[f"{prefix}obj_targets"].to(device).float()
            p = targets[f"{prefix}pos_mask"].to(device).float()
            tb = targets[f"{prefix}bbox_targets"].to(device).float()
            cm = targets[f"{prefix}center_mask"].to(device).float()

            obj_loss = self.criterion_vfl(o, t, p)
            iou_target = targets[f"{prefix}iou_targets"].to(device).float()
            p_sq = p if p.dim() == 4 else p.unsqueeze(1)
            iou_loss = F.mse_loss(
                torch.sigmoid(i) * p_sq, iou_target * p_sq, reduction="sum"
            ) / p_sq.sum().clamp(min=1)
            bbox_loss = self.criterion_eiou(
                out_dict[f"{prefix}bbox"], tb, cm, stride)

            weight = 1.0 if level == "p4" else self.p2_weight
            level_loss = obj_loss + 0.5 * iou_loss + 5.0 * bbox_loss
            total_loss += weight * level_loss
            comp[f"{prefix}obj"] = obj_loss.item()
            comp[f"{prefix}iou"] = iou_loss.item()
            comp[f"{prefix}bbox"] = bbox_loss.item()
        return total_loss, comp

    @staticmethod
    def _dfl_to_bbox(dfl_logits, stride, num_bins=16):
        B, C, H, W = dfl_logits.shape
        dfl = dfl_logits.view(B, 4, num_bins, H, W)
        bins = torch.arange(num_bins, device=dfl.device, dtype=torch.float32)
        soft = F.softmax(dfl, dim=2)
        offsets = (soft * bins[None, :, None, None]).sum(dim=2) - num_bins / 2.0
        ys, xs = torch.meshgrid(torch.arange(H, device=dfl.device, dtype=torch.float32),
                                torch.arange(W, device=dfl.device, dtype=torch.float32),
                                indexing="ij")
        xs = xs.unsqueeze(0).expand(B, -1, -1)
        ys = ys.unsqueeze(0).expand(B, -1, -1)
        dx = offsets[:, 0]; dy = offsets[:, 1]; dw = offsets[:, 2]; dh = offsets[:, 3]
        cx = (xs + 0.5 + dx) * stride
        cy = (ys + 0.5 + dy) * stride
        w = torch.exp(dw.clamp(max=5.0)) * stride
        h = torch.exp(dh.clamp(max=5.0)) * stride
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)


# ============================================================================
# ATSS Label Assignment — with gaussian fallback
# ============================================================================

def atss_assign(face_l, face_t, face_r, face_b, stride, gh, gw):
    cell_l = (np.arange(gw, dtype=np.float32)[None, :] + 0.0) * stride
    cell_t = (np.arange(gh, dtype=np.float32)[:, None] + 0.0) * stride
    cell_r = cell_l + stride; cell_b = cell_t + stride
    inter_l = np.maximum(cell_l, face_l); inter_t = np.maximum(cell_t, face_t)
    inter_r = np.minimum(cell_r, face_r); inter_b = np.minimum(cell_b, face_b)
    inter = np.maximum(0, inter_r - inter_l) * np.maximum(0, inter_b - inter_t)
    face_area = float(max(1, (face_r - face_l) * (face_b - face_t)))
    union = stride * stride + face_area - inter
    iou = inter / np.maximum(union, np.float32(1e-6))
    max_iou = iou.max()
    if max_iou <= 0:
        return np.zeros_like(iou, dtype=np.float32), np.zeros_like(iou, dtype=np.float32)
    thresh = max_iou * 0.3
    pos = iou > thresh
    if pos.sum() < 3 and max_iou > 0.05:
        thresh = max_iou * 0.1
        pos = iou > thresh
    cy = (face_t + face_b) / 2 / stride
    cx = (face_l + face_r) / 2 / stride
    sigma = max((face_r - face_l) / stride / 2, 1.0)
    r = int(np.ceil(sigma * 3))
    y0, y1 = max(0, int(cy) - r), min(gh, int(cy) + r + 1)
    x0, x1 = max(0, int(cx) - r), min(gw, int(cx) + r + 1)
    quality = np.zeros((gh, gw), dtype=np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist_sq = (yy - cy)**2 + (xx - cx)**2
    quality[y0:y1, x0:x1] = np.exp(-dist_sq / (2 * sigma**2))
    if pos.sum() < 8:
        pos = np.zeros((gh, gw), dtype=np.float32)
        pos[y0:y1, x0:x1] = (dist_sq <= (sigma * 1.5)**2).astype(np.float32)
    return pos.astype(np.float32), quality


# ============================================================================
# WIDER Face Dataset (native .txt format, dual-level targets)
# ============================================================================

class WiderDataset(Dataset):
    def __init__(self, root_dir, split="train", target_h=640, target_w=800,
                 stride=4, augment=True, min_scale=384, max_scale=800,
                 return_raw_faces=False):
        self.target_h = target_h; self.target_w = target_w
        self.stride = stride; self.augment = augment
        self.min_scale = min_scale; self.max_scale = max_scale
        self.return_raw_faces = return_raw_faces
        self.samples = []
        img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        annot_file = os.path.join(root_dir, "wider_face_split",
                                   f"wider_face_{split}_bbx_gt.txt")
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"WIDER annotation not found: {annot_file}")
        with open(annot_file) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_name = lines[i].strip(); i += 1
            if i >= len(lines) or not img_name or "/" not in img_name:
                continue
            num_faces = int(lines[i].strip()); i += 1
            img_path = os.path.join(img_dir, img_name)
            faces = []
            for _ in range(num_faces):
                if i >= len(lines): break
                parts = lines[i].strip().split(); i += 1
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    if w > 5 and h > 5:
                        faces.append({"x": x, "y": y, "w": w, "h": h})
            if os.path.exists(img_path) and faces:
                self.samples.append({"image_path": img_path, "faces": faces,
                                      "_ds_weight": 1.0, "_ds_name": "wider"})
        print(f"WiderDataset: {len(self.samples)} images ({split})")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        n_attempts = 0
        while n_attempts < 10:
            result = CrossDataset._load_sample(
                self.samples[idx], self.target_h, self.target_w,
                self.stride, self.augment, self.min_scale, self.max_scale,
                return_raw_faces=self.return_raw_faces)
            if result is not None:
                return result
            idx = random.randint(0, len(self) - 1)
            n_attempts += 1
        raise RuntimeError("Failed to load a valid image after 10 attempts")


# ============================================================================
# Cross-Dataset Loader
# ============================================================================

class CrossDataset(Dataset):
    DATASETS = {
        "mafa": {"path": "data/face/mafa", "weight": 0.5, "min_quality": 0.6},
        "fddb": {"path": "data/face/fddb", "weight": 1.0, "min_quality": 0.0},
        "ufdd": {"path": "data/face/ufdd", "weight": 0.75,"min_quality": 0.5},
        "ijbc": {"path": "data/face/ijbc", "weight": 0.5, "min_quality": 0.7},
    }
    def __init__(self, datasets=None, phase=1, target_h=640, target_w=800,
                 min_scale=384, max_scale=800,
                 stride=4, augment=True, return_raw_faces=False):
        self.target_h = target_h; self.target_w = target_w
        self.min_scale = min_scale; self.max_scale = max_scale
        self.stride = stride; self.augment = augment
        self.return_raw_faces = return_raw_faces
        self.samples = []
        if datasets is None: datasets = []
        for ds in datasets:
            info = self.DATASETS.get(ds)
            if not info: continue
            ap = os.path.join(info["path"], "annotations.json")
            if os.path.exists(ap):
                with open(ap) as f:
                    anns = json.load(f).get("annotations", [])
                anns = anns if isinstance(anns, list) else [anns]
                for a in anns:
                    if a.get("quality", 1.0) >= info["min_quality"]:
                        a["_ds_weight"] = info["weight"]
                        a["_ds_name"] = ds
                        self.samples.append(a)
            else:
                print(f"WARNING: CrossDataset({ds}): no annotations.json at {info['path']}")
        if not self.samples:
            print(f"INFO: No cross-dataset samples loaded. Using WIDER-only.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self._load_sample(
            self.samples[idx], self.target_h, self.target_w,
            self.stride, self.augment, self.min_scale, self.max_scale,
            return_raw_faces=self.return_raw_faces)

    @staticmethod
    def _load_sample(ann, target_h=640, target_w=800, stride=4,
                     augment=True, min_scale=384, max_scale=800,
                     return_raw_faces=False):
        img = cv2.imread(ann["image_path"])
        if img is None:
            return None
        h, w = img.shape[:2]
        th, tw = target_h, target_w
        scale_x = scale_y = 1.0
        crop_x = crop_y = 0
        if augment and max(h, w) > 0:
            ts = np.random.randint(min_scale, max_scale)
            s = ts / min(h, w)
            nh = int(round(h * s))
            nw = int(round(w * s))
            scale_x = nw / w; scale_y = nh / h
            img = cv2.resize(img, (nw, nh))
            if nh >= th and nw >= tw:
                crop_y = np.random.randint(0, max(1, nh - th + 1))
                crop_x = np.random.randint(0, max(1, nw - tw + 1))
                img = img[crop_y:crop_y+th, crop_x:crop_x+tw]
            else:
                pad_h = max(0, th - nh); pad_w = max(0, tw - nw)
                img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                if img_pad.shape[0] >= th and img_pad.shape[1] >= tw:
                    crop_y = np.random.randint(0, max(1, img_pad.shape[0] - th + 1))
                    crop_x = np.random.randint(0, max(1, img_pad.shape[1] - tw + 1))
                    img = img_pad[crop_y:crop_y+th, crop_x:crop_x+tw]
                else:
                    crop_x = crop_y = 0; scale_x = tw / w; scale_y = th / h
                    img = cv2.resize(img, (tw, th))
        else:
            scale_x = tw / w; scale_y = th / h
            img = cv2.resize(img, (tw, th))
        raw_faces = []
        for face in ann.get("faces", []):
            x1 = int(round(face["x"] * scale_x - crop_x))
            y1 = int(round(face["y"] * scale_y - crop_y))
            fw = int(round(face["w"] * scale_x))
            fh = int(round(face["h"] * scale_y))
            x1 = max(0, x1); y1 = max(0, y1)
            fw = min(fw, tw - x1); fh = min(fh, th - y1)
            if fw > 5 and fh > 5:
                raw_faces.append({"x": x1, "y": y1, "w": fw, "h": fh})
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        targets = CrossDataset._faces_to_targets(raw_faces, th, tw, stride)
        targets["weight"] = torch.tensor(ann.get("_ds_weight", 1.0), dtype=torch.float32)
        if return_raw_faces:
            targets["raw_faces"] = raw_faces
        return tensor, targets

    @staticmethod
    def _faces_to_targets(raw_faces, target_h, target_w, stride):
        targets = {}
        for s, level in [(4, ""), (2, "p2_")]:
            gh, gw = target_h // s, target_w // s
            pos_mask = np.zeros((gh, gw), dtype=np.float32)
            iou_targets = np.zeros_like(pos_mask)
            bbox_targets = np.zeros((4, gh, gw), dtype=np.float32)
            for face in raw_faces:
                x1, y1, fw, fh = face["x"], face["y"], face["w"], face["h"]
                pm, iou = atss_assign(x1, y1, x1+fw, y1+fh, s, gh, gw)
                pos_mask = np.maximum(pos_mask, pm)
                iou_targets = np.maximum(iou_targets, iou)
                cy = (y1 + y1 + fh) / 2; cx = (x1 + x1 + fw) / 2
                cell_y = int(cy // s); cell_x = int(cx // s)
                if 0 <= cell_y < gh and 0 <= cell_x < gw:
                    bbox_targets[0, cell_y, cell_x] = (cx - (cell_x + 0.5) * s) / s
                    bbox_targets[1, cell_y, cell_x] = (cy - (cell_y + 0.5) * s) / s
                    bbox_targets[2, cell_y, cell_x] = math.log(max(fw, 4) / s)
                    bbox_targets[3, cell_y, cell_x] = math.log(max(fh, 4) / s)
            center_mask = np.zeros((gh, gw), dtype=np.float32)
            for face in raw_faces:
                x1, y1, fw, fh = face["x"], face["y"], face["w"], face["h"]
                cy_c = (y1 + fh / 2); cx_c = (x1 + fw / 2)
                cell_y = int(cy_c // s); cell_x = int(cx_c // s)
                if 0 <= cell_y < gh and 0 <= cell_x < gw:
                    center_mask[cell_y, cell_x] = 1.0
            targets[f"{level}pos_mask"] = torch.from_numpy(pos_mask).unsqueeze(0)
            targets[f"{level}center_mask"] = torch.from_numpy(center_mask).unsqueeze(0)
            targets[f"{level}iou_targets"] = torch.from_numpy(iou_targets).unsqueeze(0)
            targets[f"{level}obj_targets"] = torch.from_numpy(pos_mask.copy()).unsqueeze(0).float()
            targets[f"{level}bbox_targets"] = torch.from_numpy(bbox_targets)
        return targets


# ============================================================================
# Mosaic Augmentation
# ============================================================================

def apply_mosaic(imgs, targets, target_h, target_w, stride=4):
    if np.random.rand() > 0.5 or imgs.size(0) < 4:
        return imgs, targets
    B = imgs.size(0)
    device = imgs.device
    mosaic = torch.zeros(3, target_h * 2, target_w * 2, device=device)
    all_raw = []
    for i in range(4):
        idx = np.random.randint(0, B)
        img = imgs[idx]
        h_off = (i // 2) * target_h
        w_off = (i % 2) * target_w
        mosaic[:, h_off:h_off+target_h, w_off:w_off+target_w] = img
        raw = targets.get("raw_faces")
        if raw and isinstance(raw, list) and len(raw) > idx and isinstance(raw[idx], list):
            for face in raw[idx]:
                all_raw.append({
                    "x": face["x"] + w_off, "y": face["y"] + h_off,
                    "w": face["w"], "h": face["h"],
                })
    crop_y = np.random.randint(0, target_h)
    crop_x = np.random.randint(0, target_w)
    mosaic = mosaic[:, crop_y:crop_y+target_h, crop_x:crop_x+target_w]
    clipped = []
    for face in all_raw:
        new_x1 = max(0, face["x"] - crop_x)
        new_x2 = min(face["x"] + face["w"], crop_x + target_w) - crop_x
        new_y1 = max(0, face["y"] - crop_y)
        new_y2 = min(face["y"] + face["h"], crop_y + target_h) - crop_y
        nw = new_x2 - new_x1
        nh = new_y2 - new_y1
        if nw > 5 and nh > 5:
            clipped.append({"x": new_x1, "y": new_y1, "w": nw, "h": nh})
    all_raw = clipped
    new_t = CrossDataset._faces_to_targets(
        [f for f in all_raw if f["w"] > 5 and f["h"] > 5],
        target_h, target_w, stride)
    new_t["weight"] = torch.tensor(1.0, dtype=torch.float32)
    if "raw_faces" in targets:
        new_t["raw_faces"] = [all_raw]
    return mosaic.unsqueeze(0), new_t


# ============================================================================
# Augmentation Helpers
# ============================================================================

def apply_hsv_jitter(t):
    if np.random.rand() < 0.2:
        return t
    B = t.size(0); dev = t.device
    gray = t.mean(dim=1, keepdim=True)
    sat = torch.rand(B, 1, 1, 1, device=dev) * 0.6 + 0.7
    bright = torch.rand(B, 1, 1, 1, device=dev) * 0.6 + 0.7
    t = gray + sat * (t - gray)
    return (t * bright).clamp(0, 1)

def apply_gridmask(t, ratio=0.3):
    if np.random.rand() > 0.25: return t
    _,_,h,w = t.shape; d = np.random.randint(20,120)
    mask = torch.ones(h,w, device=t.device)
    for y in range(0,h,d):
        for x in range(0,w,d):
            if np.random.rand() < ratio:
                mask[y:min(y+d//2,h), x:min(x+d//2,w)] = 0.0
    return t * mask[None,None,:,:]

def bbox_iou(ax, ay, aw, ah, bx, by, bw, bh):
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter + 1e-8
    return inter / union

def apply_copy_paste(imgs, targets, num_extra=5, stride=4):
    B = imgs.size(0); device = imgs.device
    C, H, W = imgs.shape[1:]
    has_raw = "raw_faces" in targets
    if not has_raw:
        return imgs, targets
    all_faces = []
    for i in range(B):
        faces = targets["raw_faces"][i]
        if isinstance(faces, list):
            for face in faces:
                all_faces.append((i, face["x"], face["y"], face["w"], face["h"]))
    if len(all_faces) < 3:
        return imgs, targets
    for i in range(B):
        raw = targets["raw_faces"][i]
        if not isinstance(raw, list):
            continue
        n_current = len(raw); n_paste = num_extra - n_current
        if n_paste <= 0:
            continue
        candidates = [f for f in all_faces if f[0] != i]
        if not candidates:
            continue
        img_np = (imgs[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for _ in range(min(n_paste, len(candidates))):
            src = candidates[random.randint(0, len(candidates) - 1)]
            sid, sx, sy, sw, sh = src
            sw = min(sw, W // 2); sh = min(sh, H // 2)
            src_np = (imgs[sid].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            patch = src_np[sy:sy+sh, sx:sx+sw].copy()
            if patch.shape[0] < 5 or patch.shape[1] < 5:
                continue
            ok = False
            for _ in range(30):
                px = random.randint(0, max(1, W - sw))
                py = random.randint(0, max(1, H - sh))
                if not any(bbox_iou(px, py, sw, sh, ef["x"], ef["y"], ef["w"], ef["h"]) > 0.3 for ef in raw):
                    ok = True; break
            if not ok:
                continue
            img_np[py:py+sh, px:px+sw] = patch
            raw.append({"x": px, "y": py, "w": sw, "h": sh})
        if True:
            imgs[i] = torch.from_numpy(img_np).float().permute(2, 0, 1).to(device) / 255.0
            new_t = CrossDataset._faces_to_targets(raw, H, W, stride)
            for k, v in new_t.items():
                targets[k][i] = v.to(device)
    return imgs, targets


# ============================================================================
# Model EMA
# ============================================================================

class ModelEMA:
    def __init__(self, model, decay=0.999, obj_bias=-2.5):
        self.decay = decay
        self.ema_model = type(model)(obj_bias=obj_bias)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.to(next(model.parameters()).device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            for ema_p, raw_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(raw_p.data, alpha=1 - self.decay)
            for ema_b, raw_b in zip(self.ema_model.buffers(), model.buffers()):
                ema_b.data.copy_(raw_b.data)
    def state_dict(self): return self.ema_model.state_dict()
    def load_state_dict(self, sd): self.ema_model.load_state_dict(sd)


# ============================================================================
# Diagnostics
# ============================================================================

def collect_diagnostics(model):
    diag = {"head_weights": {}, "head_biases": {}, "bn_stats": {},
            "gradient_per_layer": {}, "output_stats": {},
            "optimizer_lrs": {}, "dataset": {}}
    for name, param in model.named_parameters():
        key = name.replace('.', '_')
        if any(x in name for x in ['head.conv', 'head.obj', 'head.iou', 'head.bbox',
                                     'p2_head.conv', 'p2_head.obj', 'p2_head.iou', 'p2_head.bbox']):
            if 'weight' in name:
                diag['head_weights'][key] = float(param.norm(2).item())
        if 'bias' in name:
            sig = float(torch.sigmoid(param.data.mean()).item())
            diag['head_biases'][key] = {"value": float(param.data.mean().item()), "sigmoid": sig}
        if param.grad is not None:
            diag['gradient_per_layer'][key] = float(param.grad.norm(2).item())
    bn_ma, bn_va, bn_n = 0.0, 0.0, 0
    for n, buf in model.named_buffers():
        if 'running_mean' in n:
            bn_ma += buf.abs().mean().item(); bn_n += 1
        elif 'running_var' in n:
            bn_va += buf.mean().item()
    if bn_n > 0:
        diag['bn_stats'] = {"running_mean_abs_avg": bn_ma/bn_n,
                            "running_var_avg": bn_va/bn_n, "n_bn_layers": bn_n}
    return diag


def compute_output_stats(out_dict):
    stats = {}
    for k in ["obj", "iou", "p2_obj", "p2_iou"]:
        if k not in out_dict: continue
        sig = torch.sigmoid(out_dict[k])
        stats[f"{k}_sig_mean"] = float(sig.mean().item())
        stats[f"{k}_sig_max"] = float(sig.max().item())
        stats[f"{k}_sig_std"] = float(sig.std().item())
    return stats


def bn_nan_prevention(model):
    for name, buf in model.named_buffers():
        if 'running_var' in name:
            buf.data.clamp_(min=1e-4)
        buf.data.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)


# ============================================================================
# Health + Detection Pipeline Checks
# ============================================================================

@torch.no_grad()
def health_check(model, device, tag="init"):
    ok = True
    model.eval()
    zero_in = torch.zeros(1, 3, 480, 640, device=device)
    out = model(zero_in)
    for k in ["obj", "iou", "p2_obj", "p2_iou", "bbox", "p2_bbox"]:
        v = out[k]
        if not torch.isfinite(v).all():
            n, i = torch.isnan(v).sum().item(), torch.isinf(v).sum().item()
            print(f"  HEALTH [{tag}] {k}: NaN={n} Inf={i}")
            ok = False
        std = v.std().item()
        if std < 1e-5:
            print(f"  HEALTH [{tag}] {k}: COLLAPSED (std={std:.2e})")
            ok = False
        if k in ["obj", "iou", "p2_obj", "p2_iou"]:
            max_s = float(torch.sigmoid(v).max().item())
            print(f"  [{tag}] {k}: std={std:.2e} max_sig={max_s:.4f}")
    if ok:
        print(f"  [{tag}] Model health: PASS")
    return ok


@torch.no_grad()
def detection_pipeline_check(model, device, tag=""):
    peaks_found, total, tiny = 0, 0, 0
    model.eval()
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
    out = model(tensor.unsqueeze(0).to(device))
    results = {}
    for prefix, stride in [("", 4), ("p2_", 2)]:
        obj = out[f"{prefix}obj"][0,0].cpu().numpy()
        bbox = out[f"{prefix}bbox"][0].cpu().numpy()
        kernel = np.ones((3,3), dtype=np.uint8)
        peaks = (obj == cv2.dilate(obj, kernel)) & (obj > -2.5)
        pf = int(peaks.sum())
        tl, ty = 0, 0
        if peaks.any():
            ys, xs = np.where(peaks)
            for cy, cx in zip(ys, xs):
                bw = float(np.exp(np.clip(bbox[2,cy,cx], -2, 5))) * stride
                bh = float(np.exp(np.clip(bbox[3,cy,cx], -2, 5))) * stride
                tl += 1
                if bw < 5 or bh < 5:
                    ty += 1
        results[f"{prefix}peaks"] = pf
        results[f"{prefix}total"] = tl
        results[f"{prefix}tiny"] = ty
        results[f"{prefix}kept"] = tl - ty
        warn = ""
        if results[f"{prefix}kept"] == 0 and tl > 0:
            warn = " ALL BOXES TINY"
        elif results[f"{prefix}kept"] == 0 and pf == 0:
            warn = " NO PEAKS"
        lname = "P4" if prefix == "" else "P2"
        print(f"  [{tag}] {lname}: {pf} peaks, {tl} boxes, {ty} tiny, {results[f'{prefix}kept']} kept{warn}")
    return results


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate_heatmap(model, loader, device):
    model.eval()
    metrics = {"obj_loss": 0.0, "tp": 0, "fp": 0, "fn": 0, "n": 0,
               "p2_obj_loss": 0.0, "p2_tp": 0, "p2_fp": 0, "p2_fn": 0}
    for batch in tqdm(loader, desc="Val"):
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items() if isinstance(v, torch.Tensor)}
        out = model(imgs)
        for prefix, key in [("", ""), ("p2_", "p2_")]:
            obj = out[f"{key}obj"]
            ol = F.binary_cross_entropy_with_logits(obj, targets[f"{key}obj_targets"])
            metrics[f"{prefix}obj_loss"] += ol.item() if torch.isfinite(ol) else 0.0
            probs = torch.sigmoid(obj)
            gt = (targets[f"{key}obj_targets"] > 0.5).float()
            pred = (probs > 0.5).float()
            metrics[f"{prefix}tp"] += (pred * gt).sum().item()
            metrics[f"{prefix}fp"] += (pred * (1 - gt)).sum().item()
            metrics[f"{prefix}fn"] += ((1 - pred) * gt).sum().item()
        metrics["n"] += 1
    n = max(metrics["n"], 1)
    res = {}
    for prefix in ["", "p2_"]:
        lname = prefix.strip("_").upper() or "P4"
        p = metrics[f"{prefix}tp"] / max(metrics[f"{prefix}tp"] + metrics[f"{prefix}fp"], 1)
        r = metrics[f"{prefix}tp"] / max(metrics[f"{prefix}tp"] + metrics[f"{prefix}fn"], 1)
        f1 = 2 * p * r / max(p + r, 1e-8) if (p + r) > 0 else 0.0
        res[f"{prefix}obj_loss"] = metrics[f"{prefix}obj_loss"] / n
        res[f"{prefix}heatmap_f1"] = f1
        res[f"{prefix}precision"] = p
        res[f"{prefix}recall"] = r
    return res


# ============================================================================
# Main
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")

    use_amp = device.type == "cuda" and not args.no_amp
    model = FaceFCNv7_1(obj_bias=args.obj_bias, use_checkpointing=args.checkpointing).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\nParams: {total_p:,} ({total_p*4/1e6:.1f} MB FP32)")
    print(f"vs SCRFD-0.5GF (570K): {570000 - total_p:+,} smaller")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    diag_dir = os.path.join(out_dir, "v71_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    health_check(model, device, "init")
    detection_pipeline_check(model, device, "init")

    optimizer = optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if 'backbone' in n and p.requires_grad],
         'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
        {'params': [p for n,p in model.named_parameters() if 'fpn' in n and p.requires_grad],
         'lr': args.lr_fpn, 'weight_decay': args.weight_decay},
        {'params': [p for n,p in model.named_parameters()
                     if 'head' in n and 'p2_head' not in n and 'bias' not in n and p.requires_grad],
         'lr': args.lr_head, 'weight_decay': 0.0},
        {'params': [p for n,p in model.named_parameters()
                     if 'head' in n and 'p2_head' not in n and 'bias' in n and p.requires_grad],
         'lr': args.lr_head, 'weight_decay': 0.0},
        {'params': [p for n,p in model.named_parameters()
                     if 'p2_head' in n and 'bias' not in n and p.requires_grad],
         'lr': args.lr_head * 0.25, 'weight_decay': 0.05},
        {'params': [p for n,p in model.named_parameters()
                     if 'p2_head' in n and 'bias' in n and p.requires_grad],
         'lr': args.lr_head * 0.25, 'weight_decay': 0.0},
    ], lr=args.lr_head, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    ema = ModelEMA(model, decay=0.999, obj_bias=args.obj_bias)
    swa_model = None
    swa_scheduler = None

    datasets = args.datasets.split(",") if args.datasets else []
    phase = args.phase
    if phase == 0:
        phase = 1 if not datasets else (2 if "distill" not in " ".join(datasets).lower() else 3)

    use_wider = True
    if not datasets:
        use_wider = True
    elif datasets == ["wider"]:
        use_wider = True; datasets = []
    elif "wider" in datasets:
        use_wider = True; datasets = [d for d in datasets if d != "wider"]
    else:
        use_wider = False

    use_raw = not args.no_augment and args.copy_paste > 0
    if use_wider:
        wider_train = WiderDataset(args.data, "train", target_h=args.target_h, target_w=args.target_w,
                                    stride=4, augment=not args.no_augment,
                                    return_raw_faces=use_raw)
        wider_val = WiderDataset(args.data, "val", target_h=args.target_h, target_w=args.target_w,
                                  stride=4, augment=False)
    cross_train = CrossDataset(datasets=datasets, target_h=args.target_h, target_w=args.target_w,
                                stride=4, augment=not args.no_augment,
                                return_raw_faces=use_raw) if datasets else None
    cross_val = CrossDataset(datasets=datasets, target_h=args.target_h, target_w=args.target_w,
                              stride=4, augment=False) if datasets else None

    if use_wider and cross_train:
        train_dataset = ConcatDataset([wider_train, cross_train])
        val_dataset = ConcatDataset([wider_val, cross_val]) if cross_val else wider_val
    elif use_wider:
        train_dataset = wider_train; val_dataset = wider_val
    elif cross_train:
        train_dataset = cross_train; val_dataset = cross_val
    else:
        raise ValueError("No training data specified.")

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch], 0)
        keys = batch[0][1].keys()
        targets = {k: [] for k in keys}
        for b in batch:
            for k in keys:
                targets[k].append(b[1][k])
        for k in keys:
            if k == "raw_faces":
                continue
            try:
                targets[k] = torch.stack(targets[k], 0)
            except (RuntimeError, TypeError):
                targets[k] = torch.tensor(targets[k])
        return imgs, targets

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True,
                               collate_fn=_collate if use_raw else None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = V7_1Loss(varifocal_gamma=args.varifocal_gamma, p2_weight=args.p2_weight)

    n_faces = sum(len(s.get("faces",[])) for s in train_dataset.samples)
    print(f"\nTrain: {len(train_dataset)} images, {n_faces} faces, {len(train_loader)} batches/ep")
    print(f"Phase {phase}, datasets: {datasets}")
    print(f"Estimated: ~{args.epochs * 250 / 3600:.1f}h ({args.epochs} epochs)")

    metrics_path = os.path.join(out_dir, "training_metrics_v71.csv")
    csv_headers = ["epoch","train_loss","obj_loss","iou_loss","bbox_loss",
                   "p2_obj_loss","p2_iou_loss","p2_bbox_loss",
                   "val_hm_obj","val_hm_f1","val_hm_prec","val_hm_rec",
                   "val_p2_hm_obj","val_p2_hm_f1","val_p2_hm_prec","val_p2_hm_rec",
                   "det_peaks","det_boxes","det_tiny","det_kept",
                   "p2_det_peaks","p2_det_boxes","p2_det_tiny","p2_det_kept",
                   "obj_bias_val","obj_bias_sig","obj_w_l2","bbox_w_l2",
                   "p2_obj_bias_val","p2_obj_bias_sig","p2_obj_w_l2","p2_bbox_w_l2",
                   "obj_sig_max","obj_sig_mean","obj_sig_std",
                   "p2_obj_sig_max","p2_obj_sig_mean","p2_obj_sig_std",
                   "bn_mean_abs","bn_var_avg",
                   "grad_norm","lr","epoch_time_s","gpu_mem_mb","nan_skips"]
    metrics_rows = []
    best_hm_f1 = 0.0
    best_det_kept = 0
    best_val_loss = float("inf")
    step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        step = ckpt.get("step", 0)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                r = csv.reader(f); next(r, None)
                metrics_rows = [row for row in r]
        print(f"Resumed epoch {start_epoch}, step {step}, best_val_f1={best_val_f1:.4f}")
        health_check(model, device, "resume")
        detection_pipeline_check(model, device, "resume")

    hard_redistribute_done = False

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        model.train()
        epoch_losses = []; epoch_comp = defaultdict(list); nan_skips = 0
        last_grad_norm = 0.0

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.hard_redistribute > 0 and epoch >= args.hard_redistribute and not hard_redistribute_done:
            hard_redistribute_done = True
            difficulties = []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Computing difficulty"):
                    imgs, _ = batch; imgs = imgs.to(device)
                    out = model(imgs)
                    q = FaceFCNv7_1.compute_quality(out["obj"], out["iou"]).cpu()
                    for i in range(imgs.size(0)):
                        qi = q[i]
                        pk = (qi == F.max_pool2d(qi.unsqueeze(0),3,stride=1,padding=1).squeeze()) & (qi > 0.15)
                        if not pk.any():
                            difficulties.append(1.0)
                        else:
                            difficulties.append(1.0 - qi[pk].max().item())
            weights = 1.0 + np.array(difficulties)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                       sampler=sampler, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=True,
                                       collate_fn=_collate if use_raw else None)
            print(f"Hard-aware redistribution: mean weight={weights.mean():.2f}, "
                  f"min={weights.min():.2f}, max={weights.max():.2f}")

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device)
            targets = {k: v.to(device).float() for k,v in targets.items() if isinstance(v, torch.Tensor)}

            if not args.no_augment and np.random.rand() < 0.5 and args.batch_size >= 4:
                imgs, targets = apply_mosaic(imgs, targets, args.target_h, args.target_w, 4)

            if args.hsv_jitter > 0 and not args.no_augment and np.random.rand() < 0.8:
                imgs = apply_hsv_jitter(imgs)
            if args.gridmask > 0 and not args.no_augment:
                imgs = apply_gridmask(imgs)
            if not args.no_augment and args.copy_paste > 0 and np.random.rand() < 0.5:
                imgs, targets = apply_copy_paste(imgs, targets, args.copy_paste, 4)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                loss, comp = criterion(out, targets)

            sw = targets.get("weight", torch.tensor(1.0, device=device)).mean()
            loss = loss * sw / args.grad_accum
            loss = loss.float()

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
            else:
                optimizer.zero_grad()
                nan_skips += 1
                continue

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                p2_params = [p for n, p in model.named_parameters() if 'p2_head' in n and p.grad is not None]
                other_params = [p for n, p in model.named_parameters() if 'p2_head' not in n and p.grad is not None]
                if p2_params:
                    torch.nn.utils.clip_grad_norm_(p2_params, max_norm=5.0)
                if other_params:
                    gn = torch.nn.utils.clip_grad_norm_(other_params, max_norm=100.0)
                else:
                    gn = 0.0
                if torch.isfinite(torch.tensor(gn)):
                    last_grad_norm = gn
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                step += 1

                if args.ckpt_batch_interval > 0 and step % args.ckpt_batch_interval == 0:
                    torch.save({
                        "epoch": epoch, "batch": batch_idx, "step": step,
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                        "args": vars(args),
                    }, os.path.join(out_dir, "face_cnn_v71_last.pth"))

            epoch_losses.append(loss.item() * args.grad_accum)
            for k, v in comp.items():
                epoch_comp[k].append(v)
            pbar_d = {k: f"{np.mean(v):.4f}" for k, v in epoch_comp.items()}
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}", **pbar_d)

        bn_nan_prevention(model)

        n_b = max(len(epoch_losses), 1)
        train_loss_v = float(np.mean(epoch_losses))
        comp_avg = {k: float(np.mean(v)) for k, v in epoch_comp.items()}

        epoch_time = time.time() - epoch_t0
        gpu_mem = torch.cuda.max_memory_allocated(device)/1e6 if device.type == "cuda" else 0

        health_check(model, device, f"ep{epoch+1}")
        detection_pipeline_check(model, device, f"ep{epoch+1}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_res = validate_heatmap(model, val_loader, device)
        hm_f1 = val_res.get("heatmap_f1", 0.0)
        p2_hm_f1 = val_res.get("p2_heatmap_f1", 0.0)

        det_res = detection_pipeline_check(model, device, f"ep{epoch+1}")
        det_kept = det_res.get("kept", 0)
        p2_det_kept = det_res.get("p2_kept", 0)

        is_best = False
        reason = ""
        if hm_f1 > best_hm_f1 + 1e-8 and det_kept >= best_det_kept:
            best_hm_f1 = hm_f1; best_det_kept = det_kept
            is_best = True; reason = f"heatmap F1={hm_f1:.4f}"
        elif det_kept > best_det_kept + 5 and hm_f1 >= best_hm_f1 - 0.02:
            best_det_kept = det_kept
            is_best = True; reason = f"det kept={det_kept}"
        elif hm_f1 > best_hm_f1 + 5e-4 and det_kept == 0:
            print(f"  WARN: heatmap F1 improved ({best_hm_f1:.4f}→{hm_f1:.4f}) but det_kept=0")

        if is_best:
            torch.save({
                "epoch": epoch+1, "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_heatmap_f1": best_hm_f1, "best_det_kept": best_det_kept,
                "train_loss": train_loss_v, "args": vars(args),
            }, os.path.join(out_dir, "face_cnn_v71_best.pth"))
            print(f"  ★ BEST ({reason})")

        if (epoch+1) % args.ckpt_interval == 0:
            torch.save({
                "epoch": epoch+1, "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss_v,
            }, os.path.join(out_dir, f"face_cnn_v71_ep{epoch+1:03d}.pth"))

        diag = collect_diagnostics(model)
        out_stats = compute_output_stats(out)

        if (epoch+1) % args.diag_interval == 0:
            json_path = os.path.join(diag_dir, f"v71_epoch_{epoch+1:03d}.json")
            json_data = {
                "epoch": epoch+1, "train_loss": train_loss_v,
                "val": val_res, "diagnostics": diag,
                "output_stats": out_stats, "loss_components": comp_avg,
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=float)

        lr_now = swa_scheduler.get_last_lr()[0] if swa_scheduler else optimizer.param_groups[0]["lr"]
        ob_sig = diag.get("head_biases",{}).get("head_obj_bias",{}).get("sigmoid", 0)
        ob_val = diag.get("head_biases",{}).get("head_obj_bias",{}).get("value", 0)
        obj_l2 = diag.get("head_weights",{}).get("head_obj_weight", 0)
        bbox_l2 = diag.get("head_weights",{}).get("head_bbox_weight", 0)
        p2ob_sig = diag.get("head_biases",{}).get("p2_head_obj_bias",{}).get("sigmoid", 0)
        p2ob_val = diag.get("head_biases",{}).get("p2_head_obj_bias",{}).get("value", 0)
        p2obj_l2 = diag.get("head_weights",{}).get("p2_head_obj_weight", 0)
        p2bbox_l2 = diag.get("head_weights",{}).get("p2_head_bbox_weight", 0)
        bn_ma = diag.get("bn_stats",{}).get("running_mean_abs_avg", -1)
        bn_va = diag.get("bn_stats",{}).get("running_var_avg", -1)
        gn = last_grad_norm  # captured before zero_grad in training loop

        det_warn = ""
        if det_kept == 0 and det_res.get("peaks", 0) > 0: det_warn = " ⚠ DEAD BBOX"
        elif det_kept == 0 and det_res.get("peaks", 0) == 0 and p2_det_kept == 0: det_warn = " ⚠ DEAD HEADS"
        is_best_str = " ★" if is_best else ""
        nan_warn = f" NaN×{nan_skips}" if nan_skips > 0 else ""
        swa_tag = " SWA" if swa_scheduler else ""

        print(f"  e{epoch+1:3d} | loss={train_loss_v:.4f} "
               f"obj={comp_avg.get('obj',0):.4f} bbox={comp_avg.get('bbox',0):.4f} | "
              f"p2_obj={comp_avg.get('p2_obj',0):.4f} | "
              f"hmF1={hm_f1:.4f} p2_hmF1={p2_hm_f1:.4f} | "
              f"det={det_kept}/{det_res.get('peaks',0)}pk "
              f"p2={p2_det_kept}/{det_res.get('p2_peaks',0)}pk | "
              f"bias={ob_sig:.3f}({ob_val:.2f}) p2_bias={p2ob_sig:.3f}({p2ob_val:.2f}) | "
              f"L2 o={obj_l2:.1f} b={bbox_l2:.1f} p2o={p2obj_l2:.1f} p2b={p2bbox_l2:.1f} | "
              f"sig_max={out_stats.get('obj_sig_max',0):.4f} "
              f"p2_sig_max={out_stats.get('p2_obj_sig_max',0):.4f} | "
              f"BN μ={bn_ma:.2f} σ²={bn_va:.1f} | "
              f"grad={gn:.2f} | LR={lr_now:.1e} | {epoch_time:.0f}s"
              + det_warn + is_best_str + nan_warn + swa_tag)

        row = [epoch+1, f"{train_loss_v:.4f}",
               f"{comp_avg.get('obj',0):.4f}", f"{comp_avg.get('iou',0):.4f}",
               f"{comp_avg.get('bbox',0):.4f}",
               f"{comp_avg.get('p2_obj',0):.4f}", f"{comp_avg.get('p2_iou',0):.4f}",
               f"{comp_avg.get('p2_bbox',0):.4f}",
               f"{val_res.get('obj_loss',0):.4f}", f"{hm_f1:.4f}",
               f"{val_res.get('precision',0):.4f}", f"{val_res.get('recall',0):.4f}",
               f"{val_res.get('p2_obj_loss',0):.4f}", f"{p2_hm_f1:.4f}",
               f"{val_res.get('p2_precision',0):.4f}", f"{val_res.get('p2_recall',0):.4f}",
               f"{det_res.get('peaks',0)}", f"{det_res.get('total',0)}",
               f"{det_res.get('tiny',0)}", f"{det_kept}",
               f"{det_res.get('p2_peaks',0)}", f"{det_res.get('p2_total',0)}",
               f"{det_res.get('p2_tiny',0)}", f"{p2_det_kept}",
               f"{ob_val:.4f}", f"{ob_sig:.4f}",
               f"{obj_l2:.4f}", f"{bbox_l2:.4f}",
               f"{p2ob_val:.4f}", f"{p2ob_sig:.4f}",
               f"{p2obj_l2:.4f}", f"{p2bbox_l2:.4f}",
               f"{out_stats.get('obj_sig_max',0):.4f}", f"{out_stats.get('obj_sig_mean',0):.4f}",
               f"{out_stats.get('obj_sig_std',0):.4f}",
               f"{out_stats.get('p2_obj_sig_max',0):.4f}", f"{out_stats.get('p2_obj_sig_mean',0):.4f}",
               f"{out_stats.get('p2_obj_sig_std',0):.4f}",
               f"{bn_ma:.4f}", f"{bn_va:.4f}",
               f"{gn:.4f}", f"{lr_now:.8f}", f"{epoch_time:.2f}", f"{gpu_mem:.0f}",
               f"{nan_skips}"]
        metrics_rows.append(row)
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_headers)
            csv.writer(f).writerows(metrics_rows)

        if epoch < args.swa_start:
            scheduler.step()
        else:
            if swa_model is None:
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
                print(f"  SWA started at epoch {epoch+1}")
            swa_model.update_parameters(model)
            swa_scheduler.step()

    print(f"\nTraining complete. Best heatmap F1: {best_hm_f1:.4f}, Best det kept: {best_det_kept}")

    if swa_model is not None:
        print("  Running SWA BN calibration on training data...")
        swa_model.eval()
        torch.optim.swa_utils.update_bn(swa_model, train_loader)
        torch.save({
            "epoch": args.epochs, "model_state_dict": swa_model.state_dict(),
            "best_heatmap_f1": best_hm_f1, "best_det_kept": best_det_kept, "args": vars(args),
        }, os.path.join(out_dir, "face_cnn_v71_swa.pth"))
        print(f"  SWA checkpoint saved")

    torch.save({
        "epoch": args.epochs, "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "best_heatmap_f1": best_hm_f1, "best_det_kept": best_det_kept,
        "train_loss": train_loss_v, "args": vars(args),
    }, os.path.join(out_dir, "face_cnn_v71.pth"))
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceCNN v7.1 Dual-Level Training")
    parser.add_argument("--data", type=str, default="data/face/widerface")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr-backbone", type=float, default=3e-4)
    parser.add_argument("--lr-fpn", type=float, default=6e-4)
    parser.add_argument("--lr-head", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--varifocal-gamma", type=float, default=2.0)
    parser.add_argument("--obj-bias", type=float, default=-2.5)
    parser.add_argument("--p2-weight", type=float, default=0.5)
    parser.add_argument("--swa-start", type=int, default=150)
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    parser.add_argument("--min-scale", type=int, default=384)
    parser.add_argument("--target-h", type=int, default=640)
    parser.add_argument("--target-w", type=int, default=800)
    parser.add_argument("--max-scale", type=int, default=800)
    parser.add_argument("--copy-paste", type=int, default=5)
    parser.add_argument("--hsv-jitter", type=float, default=0.3)
    parser.add_argument("--gridmask", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-mosaic", action="store_true")
    parser.add_argument("--hard-redistribute", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=5)
    parser.add_argument("--ckpt-batch-interval", type=int, default=0,
                        help="Save checkpoint every N batches within epoch (0=disabled)")
    parser.add_argument("--diag-interval", type=int, default=1)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    train(args)
