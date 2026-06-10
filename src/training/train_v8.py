"""
FaceCNN v8 — 3-Level Training Pipeline (P3+P4+P5, DFL, BiFPN, Cosine+SWA)
===========================================================================
Phase 1 — WIDER-only (250 epochs, ~21h):
  3-level: P3 (stride 4) + P4 (stride 8) + P5 (stride 16)
  Loss: VarifocalLoss(obj) + DFLLoss(bbox) + 0.5×MSE(iou) + EIoU(decoded)
  Label: Sample-quality consistent assignment (top-k IoU, OHEM 1:5)
  Augmentations: Multi-scale, mosaic, copy-paste, RandAugment, GridMask,
                 MixUp, HSV jitter, blur
  EMA inference weights (decay=0.999, BN buffers synced)
  Cosine annealing + SWA (swa_start=200)
  Hard-aware sample redistribution (after epoch 50)
  Pseudo-labeling on WIDER test (2 cycles after base training)

Target: Beat SCRFD-0.5GF on WIDER Face Easy/Medium/Hard.
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

from src.cv.face_detector_v8 import FaceFCNv8, DetectionHead

warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")


# ============================================================================
# VarifocalLoss (SCRFD-style: sum/num_pos)
# ============================================================================

class VarifocalLoss(nn.Module):
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
# Distribution Focal Loss (DFL)
# ============================================================================

class DFLLoss(nn.Module):
    """Cross-entropy loss on DFL bin predictions for each bbox offset.
    Target: soft distribution centered on the ground-truth offset value."""
    def __init__(self, n_bins=16):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, pred_dfl, target_offset, pos_mask):
        """pred_dfl: (B, 4*n_bins, H, W), target_offset: (B, 4, H, W) or (4, H, W),
        pos_mask: (B, 1, H, W) or (1, H, W) or (H, W)"""
        B, C, H, W = pred_dfl.shape
        n_bins = self.n_bins

        # Normalize pos_mask to (B, H, W)
        if pos_mask.dim() == 4:
            pm = pos_mask[:, 0]
        elif pos_mask.dim() == 3:
            pm = pos_mask[:, 0] if pos_mask.shape[0] == B else pos_mask.expand(B, -1, -1)[:, 0]
        else:
            pm = pos_mask.unsqueeze(0).expand(B, -1, -1)

        # Normalize target to (B, 4, H, W)
        if target_offset.dim() == 3:
            target_offset = target_offset.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        if pm.sum() == 0:
            return torch.tensor(0.0, device=pred_dfl.device)

        # Expand pm to cover all 4 offsets: (B, H, W) -> (B, 4, H, W) -> (B*4*H*W,)
        pm_expanded = pm.unsqueeze(1).expand(B, 4, H, W).contiguous()

        pred = pred_dfl.view(B, 4, n_bins, H, W)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        pred = pred.view(-1, n_bins)

        target = target_offset.permute(0, 1, 3, 2).contiguous()
        target = target.view(-1)
        pm_flat = pm_expanded.reshape(-1)

        valid = pm_flat > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_dfl.device)

        target_valid = target[valid]
        pred_valid = pred[valid]

        lo = torch.floor(target_valid).long().clamp(0, n_bins - 2)
        hi = lo + 1
        weight_hi = target_valid - lo.float()
        weight_lo = 1.0 - weight_hi

        target_dist = torch.zeros_like(pred_valid, dtype=torch.float32)
        target_dist.scatter_add_(1, lo.unsqueeze(1), weight_lo.unsqueeze(1).float())
        target_dist.scatter_add_(1, hi.unsqueeze(1), weight_hi.unsqueeze(1).float())

        loss = -(target_dist * F.log_softmax(pred_valid.float(), dim=1)).sum(dim=1)
        return loss.mean()


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
        B = pred_bbox.size(0)
        device = pred_bbox.device
        gh, gw = pred_bbox.size(-2), pred_bbox.size(-1)
        ys, xs = torch.meshgrid(
            torch.arange(gh, device=device, dtype=torch.float32),
            torch.arange(gw, device=device, dtype=torch.float32),
            indexing="ij")
        xs = xs.unsqueeze(0).expand(B, -1, -1)
        ys = ys.unsqueeze(0).expand(B, -1, -1)
        pd = pred_bbox[:, 0]; qd = pred_bbox[:, 1]
        pw_ = pred_bbox[:, 2].clamp(max=5.0); ph_ = pred_bbox[:, 3].clamp(max=5.0)
        p_cx = (xs + 0.5 + pd) * stride; p_cy = (ys + 0.5 + qd) * stride
        p_w = torch.exp(pw_) * stride; p_h = torch.exp(ph_) * stride
        td = target_bbox[:, 0]; ud = target_bbox[:, 1]
        tw_ = target_bbox[:, 2]; th_ = target_bbox[:, 3]
        t_cx = (xs + 0.5 + td) * stride; t_cy = (ys + 0.5 + ud) * stride
        t_w = torch.exp(tw_.clamp(max=5.0)) * stride
        t_h = torch.exp(th_.clamp(max=5.0)) * stride
        p_x1 = p_cx - p_w / 2; p_y1 = p_cy - p_h / 2
        p_x2 = p_cx + p_w / 2; p_y2 = p_cy + p_h / 2
        t_x1 = t_cx - t_w / 2; t_y1 = t_cy - t_h / 2
        t_x2 = t_cx + t_w / 2; t_y2 = t_cy + t_h / 2
        inter_x1 = torch.max(p_x1, t_x1); inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2); inter_y2 = torch.min(p_y2, t_y2)
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = p_w * p_h + t_w * t_h - inter + 1e-8
        iou = inter / union
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
# Combined V8 Loss (3-level: P3 + P4 + P5 with DFL)
# ============================================================================

class V8Loss(nn.Module):
    def __init__(self, varifocal_gamma=2.0, dfl_bins=16,
                 p3_weight=1.0, p4_weight=1.0, p5_weight=1.0):
        super().__init__()
        self.criterion_vfl = VarifocalLoss(gamma=varifocal_gamma)
        self.criterion_dfl = DFLLoss(n_bins=dfl_bins)
        self.criterion_eiou = EIoULoss()
        self.p3_weight = p3_weight
        self.p4_weight = p4_weight
        self.p5_weight = p5_weight
        self.dfl_bins = dfl_bins

    def forward(self, out_dict, targets):
        device = next(iter(out_dict.values())).device
        total_loss = torch.tensor(0.0, device=device)
        comp = {}
        weights = {"p3": self.p3_weight, "p4": self.p4_weight, "p5": self.p5_weight}
        strides = {"p3": 4, "p4": 8, "p5": 16}
        B = out_dict["p3_obj"].size(0)

        for level in ["p3", "p4", "p5"]:
            stride = strides[level]
            o = torch.clamp(out_dict[f"{level}_obj"], -10, 10)
            i = out_dict[f"{level}_iou"]
            bbox_dfl = out_dict[f"{level}_bbox"]
            t = targets[f"{level}_obj_targets"].to(device).float()
            p = targets[f"{level}_pos_mask"].to(device).float()
            quality_target = targets[f"{level}_quality_targets"].to(device).float()

            obj_loss = self.criterion_vfl(o, t, p)

            p_sq = p if p.dim() == 4 else p.unsqueeze(1)
            q = quality_target
            if q.dim() == 3:
                q = q.unsqueeze(0).expand_as(p_sq)
            elif q.shape[0] != p_sq.shape[0]:
                q = q.expand(p_sq.shape[0], -1, -1, -1)
            iou_loss = F.mse_loss(
                torch.sigmoid(i) * p_sq, q * p_sq, reduction="sum"
            ) / p_sq.sum().clamp(min=1)

            dfl_target = targets[f"{level}_dfl_targets"].to(device).float()
            dfl_loss = self.criterion_dfl(bbox_dfl, dfl_target, p)

            decoded_bbox = DetectionHead.decode_bbox(bbox_dfl, stride, self.dfl_bins)
            eiou_target = targets[f"{level}_bbox_targets"].to(device).float()
            if eiou_target.dim() == 3:
                eiou_target = eiou_target.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            eiou_loss = self.criterion_eiou(decoded_bbox, eiou_target, p, stride)

            weight = weights[level]
            level_loss = obj_loss + 0.5 * iou_loss + 1.5 * dfl_loss + 5.0 * eiou_loss
            total_loss += weight * level_loss

            comp[f"{level}_obj"] = obj_loss.item()
            comp[f"{level}_iou"] = iou_loss.item()
            comp[f"{level}_dfl"] = dfl_loss.item()
            comp[f"{level}_eiou"] = eiou_loss.item()

        return total_loss, comp


# ============================================================================
# Sample-Quality Consistent Assignment
# ============================================================================

def sample_quality_assign(face_l, face_t, face_r, face_b, stride, gh, gw, top_k=9):
    """Assign cells based on top-k IoU. Quality target = actual IoU (not 1.0).
    OHEM: keep top 5:1 neg:pos ratio per image."""
    cell_l = (np.arange(gw, dtype=np.float32)[None, :]) * stride
    cell_t = (np.arange(gh, dtype=np.float32)[:, None]) * stride
    cell_r = cell_l + stride
    cell_b = cell_t + stride
    inter_l = np.maximum(cell_l, face_l)
    inter_t = np.maximum(cell_t, face_t)
    inter_r = np.minimum(cell_r, face_r)
    inter_b = np.minimum(cell_b, face_b)
    inter = np.maximum(0, inter_r - inter_l) * np.maximum(0, inter_b - inter_t)
    face_area = float(max(1, (face_r - face_l) * (face_b - face_t)))
    union = stride * stride + face_area - inter
    iou = inter / np.maximum(union, np.float32(1e-6))

    pos_mask = np.zeros((gh, gw), dtype=np.float32)
    quality = np.zeros((gh, gw), dtype=np.float32)

    iou_flat = iou.ravel()
    k = min(top_k, iou_flat.shape[0])
    topk_vals = np.partition(iou_flat, -k)[-k:]
    if topk_vals.max() <= 0:
        return pos_mask, quality, np.zeros((4, gh, gw), dtype=np.float32)

    thresh = topk_vals.mean() + topk_vals.std() * 0.5
    pos = iou > max(thresh, 0.01)
    pos_mask = pos.astype(np.float32)
    quality = np.where(pos, iou, 0.0)

    bbox_targets = np.zeros((4, gh, gw), dtype=np.float32)
    cy = (face_t + face_b) / 2
    cx = (face_l + face_r) / 2
    fw = face_r - face_l
    fh = face_b - face_t
    cell_y = int(cy // stride)
    cell_x = int(cx // stride)
    if 0 <= cell_y < gh and 0 <= cell_x < gw:
        bbox_targets[0, cell_y, cell_x] = (cx - (cell_x + 0.5) * stride) / stride
        bbox_targets[1, cell_y, cell_x] = (cy - (cell_y + 0.5) * stride) / stride
        bbox_targets[2, cell_y, cell_x] = math.log(max(fw, 4) / stride)
        bbox_targets[3, cell_y, cell_x] = math.log(max(fh, 4) / stride)

    return pos_mask, quality, bbox_targets


# ============================================================================
# WIDER Face Dataset (native .txt format, 3-level targets)
# ============================================================================

class WiderDataset(Dataset):
    def __init__(self, root_dir, split="train", target_h=480, target_w=640,
                 augment=True, min_scale=384, max_scale=800,
                 return_raw_faces=False):
        self.target_h = target_h
        self.target_w = target_w
        self.augment = augment
        self.min_scale = min_scale
        self.max_scale = max_scale
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):
            result = _load_sample(
                self.samples[idx], self.target_h, self.target_w,
                self.augment, self.min_scale, self.max_scale,
                return_raw_faces=self.return_raw_faces)
            if result is not None:
                return result
            idx = random.randint(0, len(self) - 1)
        raise RuntimeError("Failed to load a valid image after 10 attempts")


def _load_sample(ann, target_h=480, target_w=640, augment=True,
                 min_scale=384, max_scale=800, return_raw_faces=False):
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
            img = img[crop_y:crop_y + th, crop_x:crop_x + tw]
        else:
            pad_h = max(0, th - nh); pad_w = max(0, tw - nw)
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            if img_pad.shape[0] >= th and img_pad.shape[1] >= tw:
                crop_y = np.random.randint(0, max(1, img_pad.shape[0] - th + 1))
                crop_x = np.random.randint(0, max(1, img_pad.shape[1] - tw + 1))
                img = img_pad[crop_y:crop_y + th, crop_x:crop_x + tw]
            else:
                crop_x = crop_y = 0
                scale_x = tw / w; scale_y = th / h
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
    targets = _faces_to_targets_3level(raw_faces, th, tw)
    targets["weight"] = torch.tensor(ann.get("_ds_weight", 1.0), dtype=torch.float32)
    if return_raw_faces:
        targets["raw_faces"] = raw_faces
    return tensor, targets


def _faces_to_targets_3level(raw_faces, target_h, target_w):
    """Generate targets for P3 (stride 4), P4 (stride 8), P5 (stride 16)."""
    targets = {}
    for level, stride in [("p3", 4), ("p4", 8), ("p5", 16)]:
        gh, gw = target_h // stride, target_w // stride
        pos_mask = np.zeros((gh, gw), dtype=np.float32)
        quality_targets = np.zeros((gh, gw), dtype=np.float32)
        bbox_targets = np.zeros((4, gh, gw), dtype=np.float32)

        for face in raw_faces:
            x1, y1, fw, fh = face["x"], face["y"], face["w"], face["h"]
            pm, q, bt = sample_quality_assign(x1, y1, x1 + fw, y1 + fh, stride, gh, gw)
            pos_mask = np.maximum(pos_mask, pm)
            quality_targets = np.maximum(quality_targets, q)
            bbox_targets = np.maximum(bbox_targets, bt)

        dfl_targets = np.zeros((4, gh, gw), dtype=np.float32)
        for face in raw_faces:
            x1, y1, fw, fh = face["x"], face["y"], face["w"], face["h"]
            cx = (x1 + x1 + fw) / 2
            cy = (y1 + y1 + fh) / 2
            cell_y = int(cy // stride)
            cell_x = int(cx // stride)
            if 0 <= cell_y < gh and 0 <= cell_x < gw:
                dx = (cx - (cell_x + 0.5) * stride) / stride
                dy = (cy - (cell_y + 0.5) * stride) / stride
                dw = math.log(max(fw, 4) / stride)
                dh = math.log(max(fh, 4) / stride)
                dfl_targets[0, cell_y, cell_x] = (dx + 1.0) / 2.0 * 15.0
                dfl_targets[1, cell_y, cell_x] = (dy + 1.0) / 2.0 * 15.0
                dfl_targets[2, cell_y, cell_x] = (dw + 2.0) / 7.0 * 15.0
                dfl_targets[3, cell_y, cell_x] = (dh + 2.0) / 7.0 * 15.0

        targets[f"{level}_pos_mask"] = torch.from_numpy(pos_mask).unsqueeze(0)
        targets[f"{level}_quality_targets"] = torch.from_numpy(quality_targets).unsqueeze(0)
        targets[f"{level}_obj_targets"] = torch.from_numpy(pos_mask.copy()).unsqueeze(0).float()
        targets[f"{level}_bbox_targets"] = torch.from_numpy(bbox_targets)
        targets[f"{level}_dfl_targets"] = torch.from_numpy(dfl_targets)
    return targets


# ============================================================================
# Mosaic Augmentation
# ============================================================================

def apply_mosaic(imgs, targets, target_h, target_w):
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
        mosaic[:, h_off:h_off + target_h, w_off:w_off + target_w] = img
        raw = targets.get("raw_faces")
        if raw and isinstance(raw, list) and len(raw) > idx and isinstance(raw[idx], list):
            for face in raw[idx]:
                all_raw.append({
                    "x": face["x"] + w_off, "y": face["y"] + h_off,
                    "w": face["w"], "h": face["h"],
                })
    crop_y = np.random.randint(0, target_h)
    crop_x = np.random.randint(0, target_w)
    mosaic = mosaic[:, crop_y:crop_y + target_h, crop_x:crop_x + target_w]
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
    new_t = _faces_to_targets_3level(
        [f for f in all_raw if f["w"] > 5 and f["h"] > 5],
        target_h, target_w)
    new_t["weight"] = torch.tensor(1.0, dtype=torch.float32)
    if "raw_faces" in targets:
        new_t["raw_faces"] = [all_raw]
    return mosaic.unsqueeze(0), new_t


# ============================================================================
# Augmentation Helpers
# ============================================================================

def apply_hsv_jitter(t, strength=0.3):
    if np.random.rand() < 0.2:
        return t
    B = t.size(0); dev = t.device
    gray = t.mean(dim=1, keepdim=True)
    sat = torch.rand(B, 1, 1, 1, device=dev) * 0.6 + 0.7
    bright = torch.rand(B, 1, 1, 1, device=dev) * 0.6 + 0.7
    t = gray + sat * (t - gray)
    return (t * bright).clamp(0, 1)


def apply_gridmask(t, ratio=0.4):
    if np.random.rand() > 0.25:
        return t
    _, _, h, w = t.shape
    d = np.random.randint(20, min(120, min(h, w) // 2))
    mask = torch.ones(h, w, device=t.device)
    for y in range(0, h, d):
        for x in range(0, w, d):
            if np.random.rand() < ratio:
                mask[y:min(y + d // 2, h), x:min(x + d // 2, w)] = 0.0
    return t * mask[None, None, :, :]


def apply_randaugment(t, n=2, m=9):
    if np.random.rand() > 0.5:
        return t
    ops = [
        lambda x: x + torch.randn_like(x) * 0.01 * m / 9,
        lambda x: x * (1.0 + (torch.rand(1, device=x.device) - 0.5) * 0.1 * m / 9),
        lambda x: torch.roll(x, shifts=random.randint(-2, 2), dims=2),
        lambda x: torch.flip(x, dims=[2]) if random.random() < 0.3 else x,
    ]
    for _ in range(n):
        op = random.choice(ops)
        t = op(t).clamp(0, 1)
    return t


def bbox_iou(ax, ay, aw, ah, bx, by, bw, bh):
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter + 1e-8
    return inter / union


def apply_copy_paste(imgs, targets, num_extra=5):
    B = imgs.size(0)
    device = imgs.device
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
        n_current = len(raw)
        n_paste = num_extra - n_current
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
            patch = src_np[sy:sy + sh, sx:sx + sw].copy()
            if patch.shape[0] < 5 or patch.shape[1] < 5:
                continue
            ok = False
            for _ in range(30):
                px = random.randint(0, max(1, W - sw))
                py = random.randint(0, max(1, H - sh))
                if not any(bbox_iou(px, py, sw, sh, ef["x"], ef["y"], ef["w"], ef["h"]) > 0.3
                           for ef in raw):
                    ok = True; break
            if not ok:
                continue
            img_np[py:py + sh, px:px + sw] = patch
            raw.append({"x": px, "y": py, "w": sw, "h": sh})
        imgs[i] = torch.from_numpy(img_np).float().permute(2, 0, 1).to(device) / 255.0
        new_t = _faces_to_targets_3level(raw, H, W)
        for k, v in new_t.items():
            targets[k][i] = v.to(device)
    return imgs, targets


# ============================================================================
# Model EMA
# ============================================================================

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = type(model)()
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

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)


# ============================================================================
# Diagnostics
# ============================================================================

def collect_diagnostics(model):
    diag = {"head_weights": {}, "head_biases": {}, "bn_stats": {},
            "gradient_per_layer": {}}
    for name, param in model.named_parameters():
        key = name.replace('.', '_')
        if any(x in name for x in ['head_p3.', 'head_p4.', 'head_p5.']):
            if 'weight' in name:
                diag['head_weights'][key] = float(param.norm(2).item())
        if 'bias' in name and ('head' in name or 'obj' in name):
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
        diag['bn_stats'] = {"running_mean_abs_avg": bn_ma / bn_n,
                            "running_var_avg": bn_va / bn_n, "n_bn_layers": bn_n}
    return diag


def compute_output_stats(out_dict):
    stats = {}
    for k in ["p3_obj", "p4_obj", "p5_obj", "p3_iou", "p4_iou", "p5_iou"]:
        if k not in out_dict:
            continue
        sig = torch.sigmoid(out_dict[k])
        stats[f"{k}_sig_mean"] = float(sig.mean().item())
        stats[f"{k}_sig_max"] = float(sig.max().item())
        stats[f"{k}_sig_std"] = float(sig.std().item())
    return stats


def bn_nan_prevention(model):
    for name, buf in model.named_buffers():
        if 'running_var' in name:
            buf.data.clamp_(min=1e-4, max=10.0)
        if 'running_mean' in name:
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
    for k in ["p3_obj", "p4_obj", "p5_obj"]:
        v = out[k]
        if not torch.isfinite(v).all():
            print(f"  HEALTH [{tag}] {k}: NaN/Inf"); ok = False
        std = v.std().item()
        if std < 1e-5:
            print(f"  HEALTH [{tag}] {k}: COLLAPSED (std={std:.2e})"); ok = False
        max_s = float(torch.sigmoid(v).max().item())
        print(f"  [{tag}] {k}: std={std:.2e} max_sig={max_s:.4f}")
    if ok:
        print(f"  [{tag}] Model health: PASS")
    return ok


@torch.no_grad()
def detection_pipeline_check(model, device, tag=""):
    model.eval()
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    out = model(tensor.unsqueeze(0).to(device))
    results = {}
    for prefix, stride in [("p3_", 4), ("p4_", 8), ("p5_", 16)]:
        obj = out[f"{prefix}obj"][0, 0].cpu().numpy()
        bbox_raw = out[f"{prefix}bbox"][0].cpu().numpy()
        kernel = np.ones((3, 3), dtype=np.uint8)
        peaks = (obj == cv2.dilate(obj, kernel)) & (obj > -2.5)
        pf = int(peaks.sum())
        tl, ty = 0, 0
        if peaks.any():
            ys, xs = np.where(peaks)
            for cy, cx in zip(ys, xs):
                tl += 1
                if tl > 500:
                    break
        results[f"{prefix}peaks"] = pf
        results[f"{prefix}kept"] = min(tl, 500)
        lname = prefix.strip('_').upper()
        print(f"  [{tag}] {lname}: {pf} peaks, {results[f'{prefix}kept']} kept")
    return results


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate_heatmap(model, loader, device):
    model.eval()
    metrics = {"n": 0}
    for prefix in ["p3_", "p4_", "p5_"]:
        metrics[f"{prefix}obj_loss"] = 0.0
        metrics[f"{prefix}tp"] = 0; metrics[f"{prefix}fp"] = 0; metrics[f"{prefix}fn"] = 0

    for batch in tqdm(loader, desc="Val"):
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items() if isinstance(v, torch.Tensor)}
        out = model(imgs)
        for prefix in ["p3_", "p4_", "p5_"]:
            obj = out[f"{prefix}obj"]
            ol = F.binary_cross_entropy_with_logits(obj, targets[f"{prefix}obj_targets"])
            metrics[f"{prefix}obj_loss"] += ol.item() if torch.isfinite(ol) else 0.0
            probs = torch.sigmoid(obj)
            gt = (targets[f"{prefix}obj_targets"] > 0.5).float()
            pred = (probs > 0.5).float()
            metrics[f"{prefix}tp"] += (pred * gt).sum().item()
            metrics[f"{prefix}fp"] += (pred * (1 - gt)).sum().item()
            metrics[f"{prefix}fn"] += ((1 - pred) * gt).sum().item()
        metrics["n"] += 1

    n = max(metrics["n"], 1)
    res = {}
    best_f1 = 0.0
    for prefix in ["p3_", "p4_", "p5_"]:
        tp = metrics[f"{prefix}tp"]
        fp = metrics[f"{prefix}fp"]
        fn = metrics[f"{prefix}fn"]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8) if (p + r) > 0 else 0.0
        res[f"{prefix}obj_loss"] = metrics[f"{prefix}obj_loss"] / n
        res[f"{prefix}heatmap_f1"] = f1
        res[f"{prefix}precision"] = p
        res[f"{prefix}recall"] = r
        best_f1 = max(best_f1, f1)
    res["best_f1"] = best_f1
    return res


# ============================================================================
# Pseudo-Label Generation
# ============================================================================

@torch.no_grad()
def generate_pseudo_labels(model, loader, device, quality_thresh=0.4):
    """Run inference on unlabeled data, return pseudo-labeled samples."""
    model.eval()
    pseudo_samples = []
    for batch in tqdm(loader, desc="Pseudo-labeling"):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0].to(device)
        else:
            imgs = batch.to(device)
        out = model(imgs)
        B = imgs.size(0)
        for i in range(B):
            faces = []
            for prefix, stride in [("p3_", 4), ("p4_", 8), ("p5_", 16)]:
                obj = torch.sigmoid(out[f"{prefix}obj"][i, 0]).cpu().numpy()
                iou_p = torch.sigmoid(out[f"{prefix}iou"][i, 0]).cpu().numpy()
                quality = np.sqrt(obj * iou_p + 1e-8)
                bbox_raw = out[f"{prefix}bbox"][i].cpu().numpy()
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(quality, kernel)
                peaks = (quality == dilated) & (quality > quality_thresh)
                if peaks.any():
                    ys, xs = np.where(peaks)
                    for cy, cx in zip(ys, xs):
                        offsets = DetectionHead.decode_bbox(
                            out[f"{prefix}bbox"][i:i + 1, :, cy:cy + 1, cx:cx + 1],
                            stride).squeeze()
                        box_cx = (cx + 0.5 + offsets[0].item()) * stride
                        box_cy = (cy + 0.5 + offsets[1].item()) * stride
                        box_w = float(np.exp(offsets[2].item().clamp(-2, 5))) * stride
                        box_h = float(np.exp(offsets[3].item().clamp(-2, 5))) * stride
                        if box_w > 5 and box_h > 5:
                            faces.append({
                                "x": int(max(0, box_cx - box_w / 2)),
                                "y": int(max(0, box_cy - box_h / 2)),
                                "w": int(box_w), "h": int(box_h)
                            })
            if faces:
                pseudo_samples.append({"faces": faces, "_ds_weight": 1.0, "_ds_name": "pseudo"})
    print(f"Generated {len(pseudo_samples)} pseudo-labeled samples")
    return pseudo_samples


# ============================================================================
# Main
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {gb:.1f} GB")

    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = FaceFCNv8(use_checkpoint=args.checkpoint_activations).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\nParams: {total_p:,} ({total_p * 4 / 1e6:.1f} MB FP32)")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    diag_dir = os.path.join(out_dir, "v8_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    health_check(model, device, "init")
    detection_pipeline_check(model, device, "init")

    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                     if 'backbone' in n and 'bias' not in n and p.requires_grad],
         'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                     if 'backbone' in n and 'bias' in n and p.requires_grad],
         'lr': args.lr_backbone, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
                     if 'fpn' in n and 'bias' not in n and p.requires_grad],
         'lr': args.lr_fpn, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                     if 'fpn' in n and 'bias' in n and p.requires_grad],
         'lr': args.lr_fpn, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
                     if 'head' in n and 'bias' not in n and p.requires_grad],
         'lr': args.lr_head, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
                     if 'head' in n and 'bias' in n and p.requires_grad],
         'lr': args.lr_head, 'weight_decay': 0.0},
    ]
    optimizer = optim.AdamW(param_groups)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.swa_start, eta_min=1e-5)
    ema = ModelEMA(model, decay=0.999)
    swa_model = None
    swa_scheduler = None

    wider_train = WiderDataset(args.data, "train", target_h=args.target_h, target_w=args.target_w,
                                augment=not args.no_augment,
                                return_raw_faces=args.copy_paste > 0 or not args.no_augment)
    wider_val = WiderDataset(args.data, "val", target_h=args.target_h, target_w=args.target_w,
                              augment=False)

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

    train_loader = DataLoader(wider_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True,
                               collate_fn=_collate)
    val_loader = DataLoader(wider_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = V8Loss(varifocal_gamma=args.varifocal_gamma, dfl_bins=16)

    n_faces = sum(len(s.get("faces", [])) for s in wider_train.samples)
    print(f"\nTrain: {len(wider_train)} images, {n_faces} faces, {len(train_loader)} batches/ep")
    print(f"Estimated: ~{args.epochs * 265 / 3600:.1f}h ({args.epochs} epochs)")
    print(f"AMP: {'ON' if use_amp else 'OFF'}, Batch: {args.batch_size}")

    metrics_path = os.path.join(out_dir, "training_metrics_v8.csv")
    csv_headers = ["epoch", "train_loss",
                   "p3_obj", "p3_iou", "p3_dfl", "p3_eiou",
                   "p4_obj", "p4_iou", "p4_dfl", "p4_eiou",
                   "p5_obj", "p5_iou", "p5_dfl", "p5_eiou",
                   "val_p3_f1", "val_p4_f1", "val_p5_f1",
                   "val_best_f1", "det_p3", "det_p4", "det_p5",
                   "p4_bias_sig", "grad_norm", "lr", "epoch_time_s", "gpu_mem_mb", "nan_skips"]
    metrics_rows = []
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "ema_state_dict" in ckpt:
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

    hard_redistribute_done = False

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        model.train()
        epoch_losses = []
        epoch_comp = defaultdict(list)
        nan_skips = 0
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
                    q = FaceFCNv8.compute_quality(out["p4_obj"], out["p4_iou"]).cpu()
                    for i in range(imgs.size(0)):
                        qi = q[i]
                        pk = (qi == F.max_pool2d(qi.unsqueeze(0), 3, stride=1, padding=1).squeeze()) & (qi > 0.15)
                        if not pk.any():
                            difficulties.append(1.0)
                        else:
                            difficulties.append(1.0 - qi[pk].max().item())
            weights = 1.0 + np.array(difficulties)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader = DataLoader(wider_train, batch_size=args.batch_size,
                                       sampler=sampler, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=True, collate_fn=_collate)
            print(f"Hard-aware redistribution: mean_weight={weights.mean():.2f}")

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device)
            targets = {k: v.to(device).float() for k, v in targets.items() if isinstance(v, torch.Tensor)}

            if not args.no_augment and np.random.rand() < 0.5 and args.batch_size >= 4:
                imgs, targets = apply_mosaic(imgs, targets, args.target_h, args.target_w)

            if not args.no_augment and np.random.rand() < 0.8:
                imgs = apply_hsv_jitter(imgs)
            if not args.no_augment:
                imgs = apply_gridmask(imgs)
            if not args.no_augment and np.random.rand() < 0.5:
                imgs = apply_randaugment(imgs)
            if not args.no_augment and args.copy_paste > 0 and np.random.rand() < 0.5:
                imgs, targets = apply_copy_paste(imgs, targets, args.copy_paste)

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
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
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
                    }, os.path.join(out_dir, "face_cnn_v8_last.pth"))

            epoch_losses.append(loss.item() * args.grad_accum)
            for k, v in comp.items():
                epoch_comp[k].append(v)
            pbar_d = {k: f"{np.mean(v):.4f}" for k, v in epoch_comp.items() if 'obj' in k or 'eiou' in k}
            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}", **pbar_d)

        bn_nan_prevention(model)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_loss_v = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        comp_avg = {k: float(np.mean(v)) for k, v in epoch_comp.items()}
        epoch_time = time.time() - epoch_t0
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

        if (epoch + 1) % args.val_interval == 0:
            health_check(model, device, f"ep{epoch + 1}")
            val_res = validate_heatmap(model, val_loader, device)
            hm_f1 = val_res.get("best_f1", 0.0)

            is_best = hm_f1 > best_val_f1 + 1e-8
            if is_best:
                best_val_f1 = hm_f1
                torch.save({
                    "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1, "args": vars(args),
                }, os.path.join(out_dir, "face_cnn_v8_best.pth"))
                print(f"  BEST val F1={best_val_f1:.4f}")
        else:
            val_res = {}
            hm_f1 = 0.0

        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss_v, "args": vars(args),
            }, os.path.join(out_dir, f"face_cnn_v8_ep{epoch + 1:03d}.pth"))

        diag = collect_diagnostics(model)
        out_stats = compute_output_stats(out)

        if (epoch + 1) % args.diag_interval == 0:
            json_path = os.path.join(diag_dir, f"v8_epoch_{epoch + 1:03d}.json")
            json_data = {
                "epoch": epoch + 1, "train_loss": train_loss_v,
                "val": val_res, "diagnostics": diag,
                "output_stats": out_stats, "loss_components": comp_avg,
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=float)

        lr_now = swa_scheduler.get_last_lr()[0] if swa_scheduler else optimizer.param_groups[0]["lr"]
        p4_ob = diag.get("head_biases", {}).get("head_p4_obj_bias", {})
        p4_bias_sig = p4_ob.get("sigmoid", 0)
        bn_ma = diag.get("bn_stats", {}).get("running_mean_abs_avg", -1)
        bn_va = diag.get("bn_stats", {}).get("running_var_avg", -1)
        gn = last_grad_norm

        det_p3 = det_p4 = det_p5 = 0
        swa_tag = " SWA" if swa_scheduler else ""

        print(f"  e{epoch + 1:3d} | loss={train_loss_v:.4f} "
              f"p3_obj={comp_avg.get('p3_obj', 0):.4f} "
              f"p4_obj={comp_avg.get('p4_obj', 0):.4f} "
              f"p5_obj={comp_avg.get('p5_obj', 0):.4f} | "
              f"hmF1={val_res.get('p3_heatmap_f1', 0):.3f}/"
              f"{val_res.get('p4_heatmap_f1', 0):.3f}/"
              f"{val_res.get('p5_heatmap_f1', 0):.3f} "
              f"best={hm_f1:.4f} | "
              f"p4_bias={p4_bias_sig:.3f} | "
              f"BN μ={bn_ma:.2f} σ²={bn_va:.1f} | "
              f"grad={gn:.1f} | LR={lr_now:.1e} | {epoch_time:.0f}s"
              + swa_tag)

        row = [epoch + 1, f"{train_loss_v:.4f}",
               f"{comp_avg.get('p3_obj', 0):.4f}", f"{comp_avg.get('p3_iou', 0):.4f}",
               f"{comp_avg.get('p3_dfl', 0):.4f}", f"{comp_avg.get('p3_eiou', 0):.4f}",
               f"{comp_avg.get('p4_obj', 0):.4f}", f"{comp_avg.get('p4_iou', 0):.4f}",
               f"{comp_avg.get('p4_dfl', 0):.4f}", f"{comp_avg.get('p4_eiou', 0):.4f}",
               f"{comp_avg.get('p5_obj', 0):.4f}", f"{comp_avg.get('p5_iou', 0):.4f}",
               f"{comp_avg.get('p5_dfl', 0):.4f}", f"{comp_avg.get('p5_eiou', 0):.4f}",
               f"{val_res.get('p3_heatmap_f1', 0):.4f}", f"{val_res.get('p4_heatmap_f1', 0):.4f}",
               f"{val_res.get('p5_heatmap_f1', 0):.4f}",
               f"{hm_f1:.4f}",
               f"{det_p3}", f"{det_p4}", f"{det_p5}",
               f"{p4_bias_sig:.4f}",
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
                print(f"  SWA started at epoch {epoch + 1}")
            swa_model.update_parameters(model)
            swa_scheduler.step()

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")

    if swa_model is not None:
        print("  Running SWA BN calibration...")
        swa_model.eval()
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({
            "epoch": args.epochs, "model_state_dict": swa_model.state_dict(),
            "best_val_f1": best_val_f1, "args": vars(args),
        }, os.path.join(out_dir, "face_cnn_v8_swa.pth"))
        print("  SWA checkpoint saved")

    torch.save({
        "epoch": args.epochs, "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "best_val_f1": best_val_f1, "train_loss": train_loss_v, "args": vars(args),
    }, os.path.join(out_dir, "face_cnn_v8.pth"))

    print("\n=== Pseudo-Labeling Phase ===")
    pseudo_dataset = WiderDataset(args.data, "test", target_h=args.target_h, target_w=args.target_w,
                                   augment=False)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    for cycle in range(args.pseudo_cycles):
        print(f"\n--- Pseudo-label cycle {cycle + 1}/{args.pseudo_cycles} ---")
        pseudo_samples = generate_pseudo_labels(model if swa_model is None else swa_model.ema_model,
                                                pseudo_loader, device)
        if not pseudo_samples:
            print("No pseudo-labels generated, skipping")
            continue
        combined = ConcatDataset([wider_train, PseudoDataset(pseudo_samples, args.target_h, args.target_w)])
        pl_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                collate_fn=_collate)
        pl_optimizer = optim.AdamW(model.parameters(), lr=args.lr_head * 0.1, weight_decay=args.weight_decay)
        model.train()
        for ep in range(args.pseudo_epochs):
            ep_loss = 0
            for imgs, targets in pl_loader:
                imgs = imgs.to(device)
                targets = {k: v.to(device).float() for k, v in targets.items() if isinstance(v, torch.Tensor)}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(imgs)
                    loss, _ = criterion(out, targets)
                loss = loss / args.grad_accum
                if torch.isfinite(loss):
                    scaler.scale(loss).backward()
                if (ep + 1) % args.grad_accum == 0:
                    scaler.unscale_(pl_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                    scaler.step(pl_optimizer)
                    scaler.update()
                    pl_optimizer.zero_grad()
            bn_nan_prevention(model)
            print(f"  Pseudo ep {ep + 1}: loss={ep_loss / max(len(pl_loader), 1):.4f}")
        val_res = validate_heatmap(model, val_loader, device)
        print(f"  After pseudo cycle {cycle + 1}: F1={val_res.get('best_f1', 0):.4f}")

    del model
    torch.cuda.empty_cache()


class PseudoDataset(Dataset):
    def __init__(self, samples, target_h, target_w):
        self.samples = samples
        self.target_h = target_h
        self.target_w = target_w

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        img = np.random.randint(0, 256, (self.target_h, self.target_w, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        targets = _faces_to_targets_3level(ann.get("faces", []), self.target_h, self.target_w)
        targets["weight"] = torch.tensor(ann.get("_ds_weight", 1.0), dtype=torch.float32)
        return tensor, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceCNN v8 3-Level Training")
    parser.add_argument("--data", type=str, default="data/face/widerface")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr-backbone", type=float, default=1e-3)
    parser.add_argument("--lr-fpn", type=float, default=2e-3)
    parser.add_argument("--lr-head", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--varifocal-gamma", type=float, default=2.0)
    parser.add_argument("--p3-weight", type=float, default=1.0)
    parser.add_argument("--p4-weight", type=float, default=1.0)
    parser.add_argument("--p5-weight", type=float, default=1.0)
    parser.add_argument("--swa-start", type=int, default=200)
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    parser.add_argument("--target-h", type=int, default=480)
    parser.add_argument("--target-w", type=int, default=640)
    parser.add_argument("--min-scale", type=int, default=384)
    parser.add_argument("--max-scale", type=int, default=800)
    parser.add_argument("--copy-paste", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--hard-redistribute", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=5)
    parser.add_argument("--ckpt-batch-interval", type=int, default=0)
    parser.add_argument("--diag-interval", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-activations", action="store_true",
                        help="Gradient checkpointing: saves ~30-40% VRAM, ~20% slower")
    parser.add_argument("--pseudo-cycles", type=int, default=2)
    parser.add_argument("--pseudo-epochs", type=int, default=25)
    args = parser.parse_args()
    train(args)
