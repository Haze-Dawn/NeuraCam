import os, sys, time, csv, math, json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ConvergenceDetector:
    def __init__(self, window=10, eps_loss=0.001, eps_f1=0.005,
                 min_epochs=12, min_f1=0.10):
        self.window = window
        self.eps_loss = eps_loss
        self.eps_f1 = eps_f1
        self.min_epochs = min_epochs
        self.min_f1 = min_f1
        self.history = []
        self.best_val_loss = float("inf")
        self.best_f1 = 0.0
        self.converged = False
        self.reason = ""

    def update(self, epoch, val_loss, val_f1, grad_norm, wt_cos, l1_ratio):
        self.history.append({
            "epoch": epoch, "val_loss": val_loss, "val_f1": val_f1,
            "grad_norm": grad_norm, "wt_cos": wt_cos, "l1_ratio": l1_ratio,
        })
        if val_loss < self.best_val_loss - 1e-8:
            self.best_val_loss = val_loss
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
        if epoch < self.min_epochs:
            return False

        window = self.history[-self.window:] if len(self.history) >= self.window else self.history
        best_in_window = min(h["val_loss"] for h in window)
        loss_improved = (self.best_val_loss - best_in_window) > self.eps_loss
        best_f1_window = max(h["val_f1"] for h in window)
        f1_improved = (best_f1_window - self.best_f1) > self.eps_f1
        avg_grad = float(np.mean([h["grad_norm"] for h in window]))
        grad_small = avg_grad < 0.5
        avg_cos = float(np.mean([h["wt_cos"] for h in window]))
        weights_frozen = avg_cos > 0.9999
        avg_l1 = float(np.mean([h["l1_ratio"] for h in window]))
        l1_tiny = avg_l1 < 0.003
        epochs_a = np.array([h["epoch"] for h in window], dtype=np.float64)
        loss_a = np.array([h["val_loss"] for h in window], dtype=np.float64)
        if len(window) >= 3:
            slope = np.polyfit(epochs_a, loss_a, 1)[0]
        else:
            slope = 0.0
        slope_flat = abs(slope) < self.eps_loss / 10
        current_f1 = val_f1
        f1_above_floor = current_f1 >= self.min_f1
        not_improving = not loss_improved and not f1_improved
        secondary = sum([slope_flat, weights_frozen, grad_small]) >= 2
        self.converged = not_improving and secondary and l1_tiny and f1_above_floor

        if self.converged:
            signals = []
            if not loss_improved: signals.append("loss_plat")
            if not f1_improved: signals.append("f1_plat")
            if slope_flat: signals.append("slope=0")
            if weights_frozen: signals.append("wt_frozen")
            if grad_small: signals.append(f"grad={avg_grad:.2f}")
            if l1_tiny: signals.append(f"l1={l1_ratio:.4f}")
            self.reason = ", ".join(signals)
        elif not f1_above_floor and not_improving:
            self.reason = f"F1={current_f1:.4f} below min_f1={self.min_f1} -- continuing"
        return self.converged


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn.functional as F
import copy
from multiprocessing import Value

_shared_train_size = Value("i", 128)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction="mean",
                 neg_pos_ratio=10):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()

        if self.reduction == "balanced":
            pos_mask = (targets > 0.5).float()
            n_pos = pos_mask.sum().clamp(min=1)
            n_neg_to_sample = min(
                int(n_pos.item() * self.neg_pos_ratio),
                int((1 - pos_mask).sum().item())
            )
            pos_loss = (focal_loss * pos_mask).sum() / n_pos
            if n_neg_to_sample > 0:
                neg_loss_flat = (focal_loss * (1 - pos_mask)).flatten()
                neg_indices = torch.where(neg_loss_flat > 0)[0]
                if len(neg_indices) > n_neg_to_sample:
                    sampled = neg_indices[torch.randperm(
                        len(neg_indices), device=neg_indices.device)[:n_neg_to_sample]]
                    neg_loss = neg_loss_flat[sampled].mean()
                elif len(neg_indices) > 0:
                    neg_loss = neg_loss_flat.mean()
                else:
                    neg_loss = torch.tensor(0.0, device=logits.device)
            else:
                neg_loss = torch.tensor(0.0, device=logits.device)
            return pos_loss + neg_loss

        return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        with torch.no_grad():
            smooth_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return F.binary_cross_entropy_with_logits(logits, smooth_targets,
                                                  reduction=self.reduction)


class AnchorGIoULoss(nn.Module):
    def __init__(self, stride=8, anchor_scales=None):
        super().__init__()
        self.stride = stride
        if anchor_scales is None:
            anchor_scales = [1.5, 3.0, 6.0]
        self.register_buffer("anchor_scales", torch.tensor(anchor_scales, dtype=torch.float32))

    def forward(self, pred_bbox, target_bbox, pos_mask, anchor_idx):
        B = pred_bbox.size(0)
        pm = pos_mask > 0.5
        if pm.sum() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)

        actual_grid = pred_bbox.size(-1)

        pred_dx = pred_bbox[:, 0]
        pred_dy = pred_bbox[:, 1]
        pred_ls = pred_bbox[:, 2]
        target_dx = target_bbox[:, 0]
        target_dy = target_bbox[:, 1]
        target_ls = target_bbox[:, 2]

        cell_coords = self._make_cell_coords(B, actual_grid).to(pred_bbox.device)
        cx = cell_coords[:, 0]
        cy = cell_coords[:, 1]

        anchor_sizes = self.anchor_scales[anchor_idx] * self.stride

        pred_ls = torch.clamp(pred_ls, max=5.0)
        p_cx = (cx + 0.5 + pred_dx) * self.stride
        p_cy = (cy + 0.5 + pred_dy) * self.stride
        p_s = anchor_sizes * torch.exp(pred_ls)

        t_cx = (cx + 0.5 + target_dx) * self.stride
        t_cy = (cy + 0.5 + target_dy) * self.stride
        t_s = anchor_sizes * torch.exp(target_ls)

        p_x1, p_y1 = p_cx - p_s / 2, p_cy - p_s / 2
        p_x2, p_y2 = p_cx + p_s / 2, p_cy + p_s / 2
        t_x1, t_y1 = t_cx - t_s / 2, t_cy - t_s / 2
        t_x2, t_y2 = t_cx + t_s / 2, t_cy + t_s / 2

        xi = torch.clamp(torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1), min=0)
        yi = torch.clamp(torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1), min=0)
        inter = xi * yi
        p_area = p_s * p_s
        t_area = t_s * t_s
        union = p_area + t_area - inter + 1e-8
        iou = inter / union

        c_x1 = torch.min(p_x1, t_x1)
        c_y1 = torch.min(p_y1, t_y1)
        c_x2 = torch.max(p_x2, t_x2)
        c_y2 = torch.max(p_y2, t_y2)
        c_area = torch.clamp(c_x2 - c_x1, min=0) * torch.clamp(c_y2 - c_y1, min=0)
        giou = iou - (c_area - union) / (c_area + 1e-8)

        pm_float = pm.float().squeeze(1)
        giou = giou * pm_float
        giou = torch.nan_to_num(giou, nan=0.0, posinf=0.0, neginf=0.0)
        loss_per_cell = (1 - giou) * pm_float
        n_pos = pm_float.sum().clamp(min=1)
        giou_loss = loss_per_cell.sum() / n_pos
        giou_loss = torch.nan_to_num(giou_loss, nan=0.0)
        return giou_loss

    def _make_cell_coords(self, B, grid_size=None):
        if grid_size is None:
            grid_size = 16
        ys, xs = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([xs, ys], dim=0).unsqueeze(0)
        coords = coords.expand(B, -1, -1, -1).contiguous()
        return coords


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = self._copy_model(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _copy_model(self, model):
        return copy.deepcopy(model)

    def update(self, model):
        with torch.no_grad():
            for ema_p, raw_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(raw_p.data, alpha=1 - self.decay)
            for ema_b, raw_b in zip(self.ema_model.buffers(), model.buffers()):
                ema_b.data.copy_(raw_b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def flops_conv2d(cin, cout, k, h, w):
    return 2 * h * w * k * k * cin * cout

def flops_bn(c, h, w):
    return 4 * h * w * c

def flops_relu(c, h, w):
    return 1 * h * w * c

def flops_dw_conv2d(cin, k, h, w):
    return 2 * h * w * k * k * cin

def flops_pw_conv2d(cin, cout, h, w):
    return 2 * h * w * 1 * 1 * cin * cout

def flops_se(c, h, w):
    return 2 * c * (c // 16) * 2 + 2 * c

def compute_fcn_flops(input_h=128, input_w=128, use_depthwise=True, use_se=True):
    layers = []
    total = 0.0
    h, w = input_h, input_w

    c = flops_conv2d(3, 16, 5, h, w); total += c; layers.append(("block1_conv", c, f"{h}x{w}x3->16"))
    c = flops_bn(16, h, w); total += c; layers.append(("block1_bn", c, ""))
    c = flops_relu(16, h, w); total += c; layers.append(("block1_relu", c, ""))
    h, w = (h + 1) // 2, (w + 1) // 2

    c = flops_conv2d(16, 32, 3, h, w); total += c; layers.append(("block2_conv", c, f"{h}x{w}x16->32"))
    c = flops_bn(32, h, w); total += c; layers.append(("block2_bn", c, ""))
    c = flops_relu(32, h, w); total += c; layers.append(("block2_relu", c, ""))
    h, w = (h + 1) // 2, (w + 1) // 2

    if use_depthwise:
        c = flops_dw_conv2d(32, 3, h, w); total += c; layers.append(("block3_dw", c, f"{h}x{w}x32(dw)"))
        c = flops_pw_conv2d(32, 64, h, w); total += c; layers.append(("block3_pw", c, f"{h}x{w}x32->64(pw)"))
    else:
        c = flops_conv2d(32, 64, 3, h, w); total += c; layers.append(("block3_conv", c, f"{h}x{w}x32->64"))
    c = flops_bn(64, h, w); total += c; layers.append(("block3_bn", c, ""))
    c = flops_relu(64, h, w); total += c; layers.append(("block3_relu", c, ""))
    h, w = (h + 1) // 2, (w + 1) // 2

    if use_depthwise:
        c = flops_dw_conv2d(64, 3, h, w); total += c; layers.append(("block4_dw", c, f"{h}x{w}x64(dw,d=2)"))
        c = flops_pw_conv2d(64, 128, h, w); total += c; layers.append(("block4_pw", c, f"{h}x{w}x64->128(pw)"))
    else:
        c = flops_conv2d(64, 128, 3, h, w); total += c; layers.append(("block4_conv", c, f"{h}x{w}x64->128(d=2)"))
    c = flops_bn(128, h, w); total += c; layers.append(("block4_bn", c, ""))
    c = flops_relu(128, h, w); total += c; layers.append(("block4_relu", c, ""))

    c = flops_conv2d(64, 128, 1, h, w); total += c; layers.append(("skip_conv", c, f"{h}x{w}x64->128"))
    c = flops_conv2d(256, 128, 1, h, w); total += c; layers.append(("fuse_conv", c, f"{h}x{w}x256->128"))

    if use_se:
        c = flops_se(128, h, w); total += c; layers.append(("se_block", c, ""))

    head_channels = 1 + 3 * 3
    c = flops_conv2d(128, head_channels, 1, h, w); total += c; layers.append(("head_conv", c, f"{h}x{w}x128->{head_channels}"))

    return {
        "layers": layers,
        "total_flops": total,
        "total_gflops": total / 1e9,
        "input_shape": (input_h, input_w),
    }


def compute_inference_flops(frame_h=480, frame_w=640, scales=None, use_depthwise=True, use_se=True):
    if scales is None:
        sf = 1.15
        scales = [1.0 / (sf ** i) for i in range(5)]
    per_scale = []
    total = 0.0
    for s in scales:
        h, w = int(frame_h * s), int(frame_w * s)
        f = compute_fcn_flops(h, w, use_depthwise=use_depthwise, use_se=use_se)
        per_scale.append({"scale": s, "h": h, "w": w, "gflops": round(f["total_gflops"], 4)})
        total += f["total_flops"]
    return {
        "per_scale": per_scale,
        "total_flops": total,
        "total_gflops": total / 1e9,
        "frame_shape": (frame_h, frame_w),
        "n_scales": len(scales),
    }


from src.cv.face_detector_cnn import FaceFCN, ANCHOR_SCALES


def gaussian_heatmap(grid_size, center_y, center_x, sigma=1.5):
    ys, xs = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    dist2 = (ys - center_y) ** 2 + (xs - center_x) ** 2
    return np.exp(-dist2 / (2 * sigma * sigma)).astype(np.float32)


def apply_augmentation(img):
    if np.random.rand() < 0.5:
        img = np.fliplr(img).copy()
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() < 0.5:
        delta = np.random.uniform(-0.2, 0.2)
        img = np.clip(img.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.5:
        alpha = np.random.uniform(0.8, 1.2)
        img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return img


def apply_cutout(img, mask_size_ratio=0.15):
    h, w = img.shape[:2]
    mw = int(w * mask_size_ratio * np.random.uniform(0.5, 1.5))
    mh = int(h * mask_size_ratio * np.random.uniform(0.5, 1.5))
    x1 = np.random.randint(0, max(1, w - mw))
    y1 = np.random.randint(0, max(1, h - mh))
    img[y1:y1+mh, x1:x1+mw] = np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8)
    return img


def mosaic_crop(img, target_size):
    h, w = img.shape[:2]
    cx = int(np.random.uniform(w * 0.3, w * 0.7))
    cy = int(np.random.uniform(h * 0.3, h * 0.7))
    x1 = max(0, cx - target_size // 2)
    y1 = max(0, cy - target_size // 2)
    x2 = min(w, cx + target_size // 2)
    y2 = min(h, cy + target_size // 2)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < target_size or crop.shape[1] < target_size:
        crop = cv2.resize(crop, (target_size, target_size))
    return crop, x1, y1, cx, cy


def build_mosaic(img_paths, faces_list, target_size, stride, sigma):
    grid_size = target_size // stride
    mosaic = np.zeros((target_size * 2, target_size * 2, 3), dtype=np.uint8)
    heatmaps = []
    bbox_targets = []
    anchor_assign = []

    cx_off = target_size // 2
    cy_off = target_size // 2

    for qi in range(4):
        idx = qi % len(img_paths)
        img = cv2.imread(img_paths[idx])
        if img is None:
            img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            faces_list[qi] = []
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = faces_list[qi] if qi < len(faces_list) else []
        crop, ox, oy, _, _ = mosaic_crop(img, target_size)

        qx = (qi % 2) * target_size
        qy = (qi // 2) * target_size
        mosaic[qy:qy + crop.shape[0], qx:qx + crop.shape[1]] = crop[:target_size, :target_size]

        hm = np.zeros((grid_size, grid_size), dtype=np.float32)
        bt = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        aa = np.zeros((grid_size, grid_size), dtype=np.int32)

        for fx, fy, fw, fh in faces:
            face_cx = fx + fw / 2
            face_cy = fy + fh / 2
            face_size = max(fw, fh)

            global_cx = (face_cx - ox) + qx
            global_cy = (face_cy - oy) + qy

            if global_cx < 0 or global_cy < 0 or global_cx >= target_size * 2 or global_cy >= target_size * 2:
                continue

            if qi == 0:
                norm_cx = global_cx / (target_size * 2)
                norm_cy = global_cy / (target_size * 2)
            else:
                norm_cx = global_cx / (target_size * 2)
                norm_cy = global_cy / (target_size * 2)

            gx = norm_cx * grid_size
            gy = norm_cy * grid_size

            best_anchor = 0
            best_iou = 0
            for ai, ascale in enumerate(ANCHOR_SCALES):
                a_size = ascale * stride
                a_cx = 0.5
                a_cy = 0.5
                inter = max(0, min(a_size, face_size) - max(0, 0))
                iou_ = (inter * inter) / (a_size * a_size + face_size * face_size - inter * inter + 1e-8)
                if iou_ > best_iou:
                    best_iou = iou_
                    best_anchor = ai

            gx_int = int(round(gx))
            gy_int = int(round(gy))
            gx_int = np.clip(gx_int, 0, grid_size - 1)
            gy_int = np.clip(gy_int, 0, grid_size - 1)

            hm += gaussian_heatmap(grid_size, gy, gx, sigma=sigma)
            dx = gx - gx_int
            dy = gy - gy_int
            log_delta = np.log(max(face_size / (ANCHOR_SCALES[best_anchor] * stride), 0.01))
            bt[0, gy_int, gx_int] = dx
            bt[1, gy_int, gx_int] = dy
            bt[2, gy_int, gx_int] = log_delta
            aa[gy_int, gx_int] = best_anchor

        hm = np.clip(hm, 0, 1)
        heatmaps.append(hm)
        bbox_targets.append(bt)
        anchor_assign.append(aa)

    mosaic = cv2.resize(mosaic, (target_size, target_size))

    combined_hm = np.zeros((grid_size, grid_size), dtype=np.float32)
    combined_bt = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    combined_aa = np.zeros((grid_size, grid_size), dtype=np.int32)

    for qi in range(4):
        h_scale = grid_size / target_size
        for gy in range(grid_size):
            for gx in range(grid_size):
                val = heatmaps[qi][gy, gx]
                if val > combined_hm[gy, gx]:
                    combined_hm[gy, gx] = val
                    combined_bt[:, gy, gx] = bbox_targets[qi][:, gy, gx]
                    combined_aa[gy, gx] = anchor_assign[qi][gy, gx]

    return mosaic, combined_hm, combined_bt, combined_aa


def mixup_batch(batch_patches, batch_heatmaps, batch_bboxes, batch_anchors, alpha=0.2):
    if np.random.rand() < 0.5:
        return batch_patches, batch_heatmaps, batch_bboxes, batch_anchors, 1.0

    lam = np.random.beta(alpha, alpha)
    B = batch_patches.size(0)
    idx = torch.randperm(B)

    mixed_patches = lam * batch_patches + (1 - lam) * batch_patches[idx]
    mixed_heatmaps = lam * batch_heatmaps + (1 - lam) * batch_heatmaps[idx]
    mixed_bboxes = lam * batch_bboxes + (1 - lam) * batch_bboxes[idx]

    mixed_anchors = batch_anchors.clone()
    return mixed_patches, mixed_heatmaps, mixed_bboxes, mixed_anchors, lam


class WIDERFaceDataset(Dataset):
    def __init__(self, root_dir, split="train", input_size=128,
                 augment=False, stride=8, sigma=1.5, num_anchors=3):
        self.root_dir = root_dir
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
        self.augment = augment
        self.sigma = sigma
        self.num_anchors = num_anchors
        self.samples = []
        self.is_multiscale = False
        self.use_mosaic = False
        self.use_mixup = False
        self.use_cutout = False
        self.bg_ratio = 0.0

        img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        annot_file = os.path.join(root_dir, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")

        with open(annot_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if i >= len(lines) or not img_name:
                break
            if '/' not in img_name:
                continue
            num_faces = int(lines[i].strip())
            i += 1
            img_path = os.path.join(img_dir, img_name)
            if num_faces == 0:
                if i < len(lines) and '/' not in lines[i]:
                    i += 1
                continue
            faces = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    if w > 10 and h > 10:
                        faces.append((x, y, w, h))
            if faces and os.path.exists(img_path):
                self.samples.append((img_path, faces))

        print(f"Loaded {len(self.samples)} images ({split})")

    def __len__(self):
        return len(self.samples)

    def _sample_face_crop(self, img, faces, target_size):
        h, w = img.shape[:2]
        face = faces[np.random.randint(len(faces))]
        fx, fy, fw, fh = face
        cx = fx + fw / 2
        cy = fy + fh / 2
        face_size = max(fw, fh)
        crop_size = int(face_size * 3.0)
        x1 = max(0, int(cx - crop_size / 2))
        y1 = max(0, int(cy - crop_size / 2))
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        if x2 - x1 < 10 or y2 - y1 < 10:
            patch = cv2.resize(img, (target_size, target_size))
            new_cx = cx / w
            new_cy = cy / h
            new_size = face_size / max(w, h)
        else:
            patch = img[y1:y2, x1:x2]
            new_cx = (cx - x1) / (x2 - x1)
            new_cy = (cy - y1) / (y2 - y1)
            new_size = face_size / max(x2 - x1, y2 - y1)
            patch = cv2.resize(patch, (target_size, target_size))
        return patch, new_cx, new_cy, new_size

    def __getitem__(self, idx):
        if self.use_mosaic and self.augment and np.random.rand() < 0.5:
            indices = [idx] + [np.random.randint(0, len(self.samples)) for _ in range(3)]
            img_paths = [self.samples[i][0] for i in indices]
            faces_list = [self.samples[i][1] for i in indices]

            if self.is_multiscale:
                target_size = _shared_train_size.value
            else:
                target_size = self.input_size
            grid_size = target_size // self.stride

            mosaic_img, hm, bt, aa = build_mosaic(
                img_paths, faces_list, target_size, self.stride, self.sigma)

            patch_tensor = torch.from_numpy(mosaic_img).float().permute(2, 0, 1) / 255.0
            return (patch_tensor,
                    torch.from_numpy(hm).unsqueeze(0),
                    torch.from_numpy(bt),
                    torch.from_numpy(aa).long())

        img_path, faces = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if self.is_multiscale:
            target_size = _shared_train_size.value
        else:
            target_size = self.input_size
        grid_size = target_size // self.stride

        is_bg = self.augment and self.bg_ratio > 0 and np.random.rand() < self.bg_ratio
        if is_bg:
            x1 = np.random.randint(0, max(1, w - target_size))
            y1 = np.random.randint(0, max(1, h - target_size))
            patch = img[y1:y1+target_size, x1:x1+target_size]
            if patch.shape[0] != target_size or patch.shape[1] != target_size:
                patch = cv2.resize(patch, (target_size, target_size))
            if self.augment:
                patch = apply_augmentation(patch)
            patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
            return (patch_tensor,
                    torch.zeros((1, grid_size, grid_size)),
                    torch.zeros((3, grid_size, grid_size)),
                    torch.zeros((grid_size, grid_size), dtype=torch.long))

        patch, new_cx, new_cy, new_size = self._sample_face_crop(img, faces, target_size)

        if self.augment:
            patch = apply_augmentation(patch)
            if self.use_cutout:
                patch = apply_cutout(patch)

        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0

        gx = new_cx * grid_size
        gy = new_cy * grid_size
        heatmap = gaussian_heatmap(grid_size, gy, gx, sigma=self.sigma)
        gx_int = int(round(gx))
        gy_int = int(round(gy))
        gx_int = np.clip(gx_int, 0, grid_size - 1)
        gy_int = np.clip(gy_int, 0, grid_size - 1)

        best_anchor = 0
        best_iou = 0
        for ai, ascale in enumerate(ANCHOR_SCALES):
            a_size = ascale * self.stride
            face_size_px = new_size * target_size
            inter = min(a_size, face_size_px)
            union = a_size + face_size_px - inter
            iou_ = (inter * inter) / (max(union, 1)) if union > 0 else 0
            if iou_ > best_iou:
                best_iou = iou_
                best_anchor = ai

        dx = gx - gx_int
        dy = gy - gy_int
        face_size_px = new_size * target_size
        log_delta = np.log(max(face_size_px / (ANCHOR_SCALES[best_anchor] * self.stride), 0.01))

        bbox_target = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        bbox_target[0, gy_int, gx_int] = dx
        bbox_target[1, gy_int, gx_int] = dy
        bbox_target[2, gy_int, gx_int] = log_delta

        anchor_assign = np.zeros((grid_size, grid_size), dtype=np.int32)
        anchor_assign[gy_int, gx_int] = best_anchor

        return (patch_tensor,
                torch.from_numpy(heatmap).unsqueeze(0),
                torch.from_numpy(bbox_target),
                torch.from_numpy(anchor_assign).long())


def collate_anchor(batch):
    patches = torch.stack([b[0] for b in batch])
    heatmaps = torch.stack([b[1] for b in batch])
    bboxes = torch.stack([b[2] for b in batch])
    anchors = torch.stack([b[3] for b in batch])
    return patches, heatmaps, bboxes, anchors


def train_epoch(model, loader, optimizer, criterion_cls, criterion_bbox,
                device, use_giou=False, scaler=None, use_mixup=False, num_anchors=3):
    model.train()
    total_loss = total_obj = total_bbox = 0.0
    total_pos_cells = total_cells = 0
    total_batches = 0
    load_time = 0.0
    compute_time = 0.0

    for batch in tqdm(loader, desc="Train"):
        t0 = time.perf_counter()

        if len(batch) == 4:
            patches, heatmaps, bboxes, anchor_idx = batch
        else:
            patches, heatmaps, bboxes = batch
            anchor_idx = None

        patches = patches.to(device)
        heatmaps = heatmaps.to(device)
        bboxes = bboxes.to(device)

        if anchor_idx is not None:
            anchor_idx = anchor_idx.to(device)

        if use_mixup and np.random.rand() < 0.5:
            lam = np.random.beta(0.2, 0.2)
            B = patches.size(0)
            idx_shuf = torch.randperm(B)
            patches = lam * patches + (1 - lam) * patches[idx_shuf]
            heatmaps = lam * heatmaps + (1 - lam) * heatmaps[idx_shuf]
            bboxes = lam * bboxes + (1 - lam) * bboxes[idx_shuf]

        B = patches.size(0)
        t1 = time.perf_counter()
        load_time += t1 - t0

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(patches)
            pred_obj = outputs[:, 0:1]
            pred_bbox_raw = outputs[:, 1:]

            obj_loss = criterion_cls(pred_obj, heatmaps)
            pos_mask = (heatmaps > 0.5).float()
            pos_count = pos_mask.sum().item()
            total_pos_cells += pos_count
            gs = heatmaps.size(-1)
            total_cells += B * gs * gs

            bbox_loss = torch.tensor(0.0, device=device)
            if pos_mask.sum() > 0 and use_giou:
                for ai in range(num_anchors):
                    offset = ai * 3
                    ai_mask = heatmaps.new_zeros(heatmaps.shape)
                    if anchor_idx is not None:
                        ai_mask = ((anchor_idx == ai).float().unsqueeze(1) * pos_mask)
                    else:
                        ai_mask = pos_mask
                    if ai_mask.sum() > 0:
                        pred_ai = pred_bbox_raw[:, offset:offset+3]
                        if anchor_idx is not None:
                            target_ai = bboxes
                        else:
                            target_ai = bboxes
                        bbox_loss += criterion_bbox(pred_ai, target_ai, ai_mask, ai)
            elif pos_mask.sum() > 0:
                bbox_loss = F.smooth_l1_loss(
                    pred_bbox_raw * pos_mask.repeat(1, pred_bbox_raw.size(1), 1, 1),
                    bboxes.repeat(1, 1, 1, 1) if bboxes.size(1) == 3 else bboxes * pos_mask.repeat(1, bboxes.size(1), 1, 1),
                    reduction="sum"
                ) / (pos_mask.sum() + 1) * 5.0

            loss = obj_loss + bbox_loss

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            total_batches += 1
            compute_time += time.perf_counter() - t1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        total_obj += obj_loss.item()
        total_bbox += bbox_loss.item() if isinstance(bbox_loss, torch.Tensor) else 0
        total_batches += 1
        compute_time += time.perf_counter() - t1

    n = total_batches
    load_pct = load_time / max(load_time + compute_time, 1e-8) * 100
    return (total_loss / n, total_obj / n, total_bbox / n,
            total_pos_cells / max(total_cells, 1),
            load_time, compute_time, load_pct)


@torch.no_grad()
def validate(model, loader, criterion_cls, criterion_bbox, device, use_giou=False, num_anchors=3):
    model.eval()
    val_obj_loss = 0.0
    val_bbox_loss = 0.0
    total_pos_cells = total_cells = 0
    tp = fp = fn = tn = 0
    n_batches = 0
    total_bbox_cells = 0
    sum_dx_err = sum_dy_err = sum_logsize_err = 0.0
    batch_ious = []
    iou_05_count = iou_05_total = 0
    n_bins = 10
    bin_confidences = [0.0] * n_bins
    bin_accuracies = [0.0] * n_bins
    bin_counts = [0] * n_bins
    n_bins20 = 20
    bin20_counts = [0] * n_bins20
    bin20_confidences = [0.0] * n_bins20

    for batch in tqdm(loader, desc="Val"):
        if len(batch) == 4:
            patches, heatmaps, bboxes, anchor_idx = batch
        else:
            patches, heatmaps, bboxes = batch
            anchor_idx = None

        patches = patches.to(device)
        heatmaps = heatmaps.to(device)
        bboxes = bboxes.to(device)
        if anchor_idx is not None:
            anchor_idx = anchor_idx.to(device)
        B = patches.size(0)

        outputs = model(patches)
        pred_obj = outputs[:, 0:1]
        pred_bbox_raw = outputs[:, 1:]

        val_obj_loss += criterion_cls(pred_obj, heatmaps).item()
        pos_mask = (heatmaps > 0.5).float()
        pos_count = pos_mask.sum().item()
        total_pos_cells += pos_count
        gs = heatmaps.size(-1)
        total_cells += B * gs * gs

        if pos_mask.sum() > 0 and use_giou:
            for ai in range(num_anchors):
                offset = ai * 3
                ai_mask = heatmaps.new_zeros(heatmaps.shape)
                if anchor_idx is not None:
                    ai_mask = ((anchor_idx == ai).float().unsqueeze(1) * pos_mask)
                else:
                    ai_mask = pos_mask
                if ai_mask.sum() > 0:
                    pred_ai = pred_bbox_raw[:, offset:offset+3]
                    bbox_l = criterion_bbox(pred_ai, bboxes, ai_mask, ai)
                    val_bbox_loss += bbox_l.item()
        elif pos_mask.sum() > 0:
            bbox_l = F.smooth_l1_loss(
                pred_bbox_raw * pos_mask.repeat(1, pred_bbox_raw.size(1), 1, 1),
                bboxes * pos_mask.repeat(1, bboxes.size(1), 1, 1),
                reduction="sum"
            )
            val_bbox_loss += (bbox_l / (pos_mask.sum() + 1) * 5.0).item()

        per_cell_ious = []
        for b in range(B):
            pm = pos_mask[b, 0]
            if pm.sum() == 0:
                continue
            cell_indices = torch.where(pm > 0.5)
            for cy, cx in zip(cell_indices[0], cell_indices[1]):
                ai = 0
                if anchor_idx is not None:
                    ai = int(anchor_idx[b, cy, cx].item())
                offset = ai * 3
                p_dx = pred_bbox_raw[b, offset, cy, cx].item()
                p_dy = pred_bbox_raw[b, offset + 1, cy, cx].item()
                p_ls = pred_bbox_raw[b, offset + 2, cy, cx].item()
                t_dx = bboxes[b, 0, cy, cx].item()
                t_dy = bboxes[b, 1, cy, cx].item()
                t_ls = bboxes[b, 2, cy, cx].item()
                sum_dx_err += abs(p_dx - t_dx)
                sum_dy_err += abs(p_dy - t_dy)
                sum_logsize_err += abs(p_ls - t_ls)
                total_bbox_cells += 1
                stride = 8
                a_size = ANCHOR_SCALES[ai] * stride
                p_ls_clamped = min(p_ls, 5.0)
                t_ls_clamped = min(t_ls, 5.0)
                p_cx = (cx + 0.5 + p_dx) * stride
                p_cy = (cy + 0.5 + p_dy) * stride
                p_s = a_size * np.exp(p_ls_clamped)
                t_cx = (cx + 0.5 + t_dx) * stride
                t_cy = (cy + 0.5 + t_dy) * stride
                t_s = a_size * np.exp(t_ls_clamped)
                p_x1, p_y1 = p_cx - p_s / 2, p_cy - p_s / 2
                p_x2, p_y2 = p_cx + p_s / 2, p_cy + p_s / 2
                t_x1, t_y1 = t_cx - t_s / 2, t_cy - t_s / 2
                t_x2, t_y2 = t_cx + t_s / 2, t_cy + t_s / 2
                xi = max(0, min(p_x2, t_x2) - max(p_x1, t_x1))
                yi = max(0, min(p_y2, t_y2) - max(p_y1, t_y1))
                inter = xi * yi
                union = (p_s * p_s) + (t_s * t_s) - inter
                iou = inter / max(union, 1e-8)
                per_cell_ious.append(iou)

        if per_cell_ious:
            ious_cpu = [float(i) for i in per_cell_ious]
            batch_ious.extend(ious_cpu)
            for iou in ious_cpu:
                iou_05_total += 1
                if iou > 0.5:
                    iou_05_count += 1

        pred_probs = torch.sigmoid(pred_obj)
        pred_bin = (pred_probs > 0.5).float()
        gt_bin = (heatmaps > 0.5).float()
        tp += (pred_bin * gt_bin).sum().item()
        fp += (pred_bin * (1 - gt_bin)).sum().item()
        fn += ((1 - pred_bin) * gt_bin).sum().item()
        tn += ((1 - pred_bin) * (1 - gt_bin)).sum().item()
        gt_labels = (heatmaps > 0.5).int()
        for b in range(B):
            confs = pred_probs[b, 0].flatten()
            gts = gt_labels[b, 0].flatten()
            for c_val, g_val in zip(confs, gts):
                cv, gv = c_val.item(), g_val.item()
                if not math.isfinite(cv):
                    continue
                bin_idx = min(int(cv * n_bins), n_bins - 1)
                bin_confidences[bin_idx] += cv
                bin_accuracies[bin_idx] += gv
                bin_counts[bin_idx] += 1
                bin20_idx = min(int(cv * n_bins20), n_bins20 - 1)
                bin20_counts[bin_idx] += 1
                bin20_confidences[bin20_idx] += cv

        n_batches += 1

    val_obj_loss /= n_batches
    val_bbox_loss /= n_batches
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)
    pos_ratio = total_pos_cells / max(total_cells, 1)
    mean_dx_err = sum_dx_err / max(total_bbox_cells, 1)
    mean_dy_err = sum_dy_err / max(total_bbox_cells, 1)
    mean_ls_err = sum_logsize_err / max(total_bbox_cells, 1)
    mean_iou = np.mean(batch_ious) if batch_ious else 0.0
    iou_at_05 = iou_05_count / max(iou_05_total, 1)
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_conf = bin_confidences[i] / bin_counts[i]
            avg_acc = bin_accuracies[i] / bin_counts[i]
            ece += (bin_counts[i] / max(sum(bin_counts), 1)) * abs(avg_conf - avg_acc)

    bin20_centers = [(i + 0.5) / n_bins20 for i in range(n_bins20)]
    total_20 = max(sum(bin20_counts), 1)
    bin20_norm = [c / total_20 for c in bin20_counts]
    bin20_avg_conf = [(bin20_confidences[i] / max(bin20_counts[i], 1)) for i in range(n_bins20)]

    return {
        "val_obj_loss": val_obj_loss,
        "val_bbox_loss": val_bbox_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "pos_ratio": pos_ratio,
        "mean_dx_err": mean_dx_err,
        "mean_dy_err": mean_dy_err,
        "mean_ls_err": mean_ls_err,
        "mean_iou": mean_iou,
        "iou_at_05": iou_at_05,
        "ece": ece,
        "confidence_20_bins": {
            "bin_centers": bin20_centers,
            "bin_counts": bin20_norm,
            "bin_avg_confidences": bin20_avg_conf,
            "mean_confidence": float(np.mean(
                [c / max(cnt, 1) for c, cnt in zip(bin20_confidences, bin20_counts)]
            )),
        },
    }


def layer_weight_stats(model):
    stats = {}
    total_weights = 0
    total_near_zero = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.data.cpu().numpy().flatten()
            w_mat = param.data.cpu().numpy().reshape(param.shape[0], -1)
            try:
                sv = np.linalg.svd(w_mat, compute_uv=False)
                spectral_norm = float(sv[0])
                s_norm = sv / max(sv.sum(), 1e-10)
                entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
                eff_rank = float(np.exp(entropy))
                smallest_sv = float(sv[-1])
            except np.linalg.LinAlgError:
                spectral_norm = 0.0
                eff_rank = 0.0
                smallest_sv = 1.0
            stats_layer = {
                "mean": float(w.mean()),
                "std": float(w.std()),
                "near_zero_pct": float((np.abs(w) < 0.01).mean() * 100),
                "shape": list(param.shape),
                "norm": float(np.linalg.norm(w)),
                "min": float(w.min()),
                "max": float(w.max()),
                "spectral_norm": spectral_norm,
                "effective_rank": eff_rank,
                "cond_number": spectral_norm / max(smallest_sv, 1e-10),
            }
            stats[name] = stats_layer
            total_weights += len(w)
            total_near_zero += (np.abs(w) < 0.01).sum()
    stats["_total"] = {
        "total_params": total_weights,
        "near_zero_pct": float(total_near_zero / max(total_weights, 1) * 100),
    }
    return stats


def layer_gradient_stats(model):
    stats = {}
    total_norm = 0.0
    max_update_ratio = 0.0
    min_update_ratio = float("inf")
    for name, param in model.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            wn = param.data.norm().item()
            stats[name] = {
                "grad_norm": gn,
                "weight_norm": wn,
                "update_ratio": gn / max(wn, 1e-10),
            }
            total_norm += gn ** 2
            max_update_ratio = max(max_update_ratio, stats[name]["update_ratio"])
            if stats[name]["update_ratio"] > 0:
                min_update_ratio = min(min_update_ratio, stats[name]["update_ratio"])
        else:
            stats[name] = {"grad_norm": 0.0, "weight_norm": 0.0, "update_ratio": 0.0}
    stats["_total_norm"] = math.sqrt(total_norm)
    stats["_max_update_ratio"] = max_update_ratio
    stats["_min_update_ratio"] = min_update_ratio if min_update_ratio != float("inf") else 0.0
    return stats


@torch.no_grad()
def activation_stats(model, loader, device):
    model.eval()
    activations = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            act = out.detach().cpu().numpy()
            if name not in activations:
                activations[name] = []
            activations[name].append(act)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(make_hook(name)))

    patches, _, _, _ = next(iter(loader))
    patches = patches.to(device)
    _ = model(patches)

    for h in handles:
        h.remove()

    stats = {}
    for name, acts_list in activations.items():
        acts = np.concatenate(acts_list, axis=0)
        dead_pct = float((acts == 0).mean() * 100)
        stats[name] = {
            "mean_activation": float(acts.mean()),
            "std_activation": float(acts.std()),
            "dead_neuron_pct": dead_pct,
            "max_activation": float(acts.max()),
        }
    return stats


def bn_statistics(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            rm = module.running_mean.cpu().numpy()
            rv = module.running_var.cpu().numpy()
            stats[name] = {
                "running_mean_mean": float(rm.mean()),
                "running_mean_std": float(rm.std()),
                "running_var_mean": float(rv.mean()),
                "running_var_std": float(rv.std()),
                "num_features": int(rm.shape[0]),
            }
    return stats


def weight_cosine_similarity(prev_dict, curr_dict):
    sims = {}
    for name in prev_dict:
        if name in curr_dict and "weight" in name:
            p = prev_dict[name].flatten()
            c = curr_dict[name].flatten()
            dot = np.dot(p, c)
            norm = np.linalg.norm(p) * np.linalg.norm(c)
            sims[name] = float(dot / max(norm, 1e-10))
    if sims:
        sims["_mean"] = float(np.mean(list(sims.values())))
    return sims


class FullFrameMineDataset(Dataset):
    def __init__(self, root_dir, split="train", input_size=128, stride=8,
                 samples_per_image=10, max_crop_attempts=50, num_anchors=3):
        self.root_dir = root_dir
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
        self.samples_per_image = samples_per_image
        self.max_crop_attempts = max_crop_attempts
        self.num_anchors = num_anchors
        self.samples = []

        img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        annot_file = os.path.join(root_dir, "wider_face_split",
                                  f"wider_face_{split}_bbx_gt.txt")
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")

        with open(annot_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if i >= len(lines) or not img_name:
                break
            if '/' not in img_name:
                continue
            num_faces = int(lines[i].strip())
            i += 1
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                for _ in range(num_faces):
                    i += 1
                continue
            faces = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    if w > 5 and h > 5:
                        faces.append((x, y, w, h))
            self.samples.append((img_path, faces))
            if num_faces == 0:
                self.samples.append((img_path, []))

        full_size = len(self.samples)
        effective_size = full_size * samples_per_image
        print(f"FullFrameMineDataset: {full_size} images, "
              f"~{effective_size} effective crops ({split})")

    def __len__(self):
        return len(self.samples) * self.samples_per_image

    def _crop_avoids_faces(self, img_h, img_w, crop_size, faces):
        for _ in range(self.max_crop_attempts):
            x1 = np.random.randint(0, max(1, img_w - crop_size))
            y1 = np.random.randint(0, max(1, img_h - crop_size))
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            ok = True
            for fx, fy, fw, fh in faces:
                ix1 = max(x1, fx)
                iy1 = max(y1, fy)
                ix2 = min(x2, fx + fw)
                iy2 = min(y2, fy + fh)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = (ix2 - ix1) * (iy2 - iy1)
                    face_area = fw * fh
                    crop_area = crop_size * crop_size
                    if overlap > 0.3 * min(face_area, crop_area):
                        ok = False
                        break
            if ok:
                return (y1, y2, x1, x2)
        return (0, min(crop_size, img_h), 0, min(crop_size, img_w))

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        img_path, faces = self.samples[img_idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        crop_size = self.input_size
        if h < crop_size or w < crop_size:
            img = cv2.resize(img, (max(w, crop_size), max(h, crop_size)))
            h, w = img.shape[:2]
        y1, y2, x1, x2 = self._crop_avoids_faces(h, w, crop_size, faces)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] != crop_size or patch.shape[1] != crop_size:
            patch = cv2.resize(patch, (crop_size, crop_size))
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
        heatmap = torch.zeros((1, self.grid_size, self.grid_size))
        bbox_target = torch.zeros((3, self.grid_size, self.grid_size))
        anchor_assign = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        return patch_tensor, heatmap, bbox_target, anchor_assign


class WiderFaceFPNDataset(Dataset):
    def __init__(self, root_dir, split="train", target_h=480, target_w=640,
                 augment=False, num_anchors=3, use_atss=False, atss_levels=None):
        self.root_dir = root_dir
        self.split = split
        self.target_h = target_h
        self.target_w = target_w
        self.augment = augment
        self.is_multiscale = False
        self.use_atss = use_atss
        self.atss_levels = atss_levels or ["p3"]
        self.samples = []

        img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        annot_file = os.path.join(root_dir, "wider_face_split",
                                  f"wider_face_{split}_bbx_gt.txt")
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")

        with open(annot_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if i >= len(lines) or not img_name:
                break
            if "/" not in img_name:
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
                    if w > 5 and h > 5:
                        faces.append((x, y, w, h))
            if os.path.exists(img_path):
                self.samples.append((img_path, faces))

        print(f"WiderFaceFPNDataset: {len(self.samples)} images ({split})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _atss_assign(face_l, face_t, face_r, face_b, stride, gh, gw):
        """ATSS: Assign positive cells based on IoU with face box.
        Returns boolean mask of shape (gh, gw).
        """
        cell_l = (np.arange(gw)[None, :] + 0.0) * stride
        cell_t = (np.arange(gh)[:, None] + 0.0) * stride
        cell_r = cell_l + stride
        cell_b = cell_t + stride

        inter_l = np.maximum(cell_l, face_l)
        inter_t = np.maximum(cell_t, face_t)
        inter_r = np.minimum(cell_r, face_r)
        inter_b = np.minimum(cell_b, face_b)
        inter = np.maximum(0, inter_r - inter_l) * np.maximum(0, inter_b - inter_t)

        cell_area = stride * stride
        face_area = (face_r - face_l) * (face_b - face_t)
        union = cell_area + face_area - inter
        iou = inter / np.maximum(union, 1e-8)

        k = min(9, gh * gw)
        flat = iou.ravel()
        topk_idx = np.argpartition(flat, -k)[-k:]
        topk_vals = flat[topk_idx]
        thresh = topk_vals.mean() + max(topk_vals.std(), 1e-8)
        pos_mask = iou > thresh
        return pos_mask, iou

    @staticmethod
    def _gaussian_heatmap(h, w, cy, cx, sigma):
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
        return np.exp(-dist2 / (2 * sigma * sigma + 1e-8)).astype(np.float32)

    @staticmethod
    def _apply_color_jitter(img):
        """Random brightness, contrast, saturation, hue. Applied in-place after targets generated."""
        if np.random.rand() < 0.5:
            return img
        img = img.astype(np.float32)
        if np.random.rand() < 0.8:
            img += np.random.uniform(-25, 25)
        if np.random.rand() < 0.8:
            img = (img - 128) * np.random.uniform(0.7, 1.3) + 128
        img = img.clip(0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if np.random.rand() < 0.8:
            hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
        if np.random.rand() < 0.5:
            hsv[:, :, 0] += np.random.uniform(-10, 10)
        hsv[:, :, 0] = hsv[:, :, 0] % 180
        hsv = hsv.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _apply_gaussian_noise(img):
        if np.random.rand() < 0.3:
            noise = np.random.randn(*img.shape).astype(np.float32) * 3
            img = img.astype(np.float32) + noise
            img = img.clip(0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _apply_cutout(img):
        if np.random.rand() < 0.5:
            return img
        h, w = img.shape[:2]
        mw = int(w * 0.2 * np.random.uniform(0.5, 1.5))
        mh = int(h * 0.2 * np.random.uniform(0.5, 1.5))
        x1 = np.random.randint(0, max(1, w - mw))
        y1 = np.random.randint(0, max(1, h - mh))
        img[y1:y1+mh, x1:x1+mw] = np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        img_path, faces = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_w, self.target_h))

        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h

        if self.augment and np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            flip_x = True
        else:
            flip_x = False

        h, w = self.target_h, self.target_w
        fpn_levels = [("p2", 2), ("p3", 4), ("p4", 8)]

        heatmaps = {}
        bboxes = {}
        for name, stride in fpn_levels:
            heatmaps[name] = np.zeros((h // stride, w // stride), dtype=np.float32)
            bboxes[name] = np.zeros((4, h // stride, w // stride), dtype=np.float32)

        for fx, fy, fw, fh in faces:
            if fw < 5 or fh < 5:
                continue
            cx = (fx + fw / 2) * scale_x
            cy = (fy + fh / 2) * scale_y
            if flip_x:
                cx = w - cx
            face_w = fw * scale_x
            face_h = fh * scale_y
            face_size = max(face_w, face_h)

            best_name = "p2"
            best_dist = float("inf")
            for name, stride in fpn_levels:
                dist = abs(face_size - stride * 4)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            stride = {"p2": 2, "p3": 4, "p4": 8}[best_name]
            gh, gw = h // stride, w // stride
            grid_cx = cx / stride
            grid_cy = cy / stride
            sigma = max(face_size / (stride * 2), 0.5)

            if self.use_atss and best_name in self.atss_levels:
                face_l = max(0, cx - face_w / 2)
                face_t = max(0, cy - face_h / 2)
                face_r = min(w, cx + face_w / 2)
                face_b = min(h, cy + face_h / 2)
                if face_r <= face_l or face_b <= face_t:
                    continue
                pos_mask, _ = self._atss_assign(face_l, face_t, face_r, face_b,
                                                  stride, gh, gw)
                heatmap_val = pos_mask.astype(np.float32)
                heatmaps[best_name] = np.maximum(heatmaps[best_name], heatmap_val)
                for gy, gx in zip(*np.where(pos_mask)):
                    dx = grid_cx - gx
                    dy = grid_cy - gy
                    dw = np.log(max(face_w / stride, 0.01))
                    dh = np.log(max(face_h / stride, 0.01))
                    bboxes[best_name][0, gy, gx] = dx
                    bboxes[best_name][1, gy, gx] = dy
                    bboxes[best_name][2, gy, gx] = dw
                    bboxes[best_name][3, gy, gx] = dh
            else:
                hm = self._gaussian_heatmap(gh, gw, grid_cy, grid_cx, sigma)
                heatmaps[best_name] = np.maximum(heatmaps[best_name], hm)

                gx = int(round(grid_cx))
                gy = int(round(grid_cy))
                gx = np.clip(gx, 0, gw - 1)
                gy = np.clip(gy, 0, gh - 1)

                dx = grid_cx - gx
                dy = grid_cy - gy
                dw = np.log(max(face_w / stride, 0.01))
                dh = np.log(max(face_h / stride, 0.01))

                bboxes[best_name][0, gy, gx] = dx
                bboxes[best_name][1, gy, gx] = dy
                bboxes[best_name][2, gy, gx] = dw
                bboxes[best_name][3, gy, gx] = dh

        if self.augment:
            img = self._apply_color_jitter(img)
            img = self._apply_gaussian_noise(img)
            img = self._apply_cutout(img)

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return (
            img_tensor,
            torch.from_numpy(heatmaps["p2"]).unsqueeze(0),
            torch.from_numpy(bboxes["p2"]),
            torch.from_numpy(heatmaps["p3"]).unsqueeze(0),
            torch.from_numpy(bboxes["p3"]),
            torch.from_numpy(heatmaps["p4"]).unsqueeze(0),
            torch.from_numpy(bboxes["p4"]),
        )


def train_epoch_v5(model, loader, optimizer, criterion_cls, criterion_bbox,
                   device, scaler=None):
    model.train()
    total_loss = 0.0
    total_obj = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    total_bbox = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    total_batches = 0
    grad_norm_sum = 0.0

    fpn_levels = ["p2", "p3", "p4"]
    level_strides = {"p2": 2, "p3": 4, "p4": 8}

    for batch in tqdm(loader, desc="Train v5"):
        (patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4) = batch
        patches = patches.to(device)
        targets = {
            "p2": (hm_p2.to(device), bb_p2.to(device)),
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(patches)
            loss = torch.tensor(0.0, device=device)
            for level in fpn_levels:
                pred_obj = outputs[f"{level}_obj"]
                pred_bbox = outputs[f"{level}_bbox"]
                t_hm, t_bb = targets[level]
                stride = level_strides[level]
                pos_mask = (t_hm > 0.5).float()

                obj_l = criterion_cls(pred_obj, t_hm)
                total_obj[level] += obj_l.item()

                bbox_l = torch.tensor(0.0, device=device)
                if pos_mask.sum() > 0:
                    bbox_l = criterion_bbox(pred_bbox, t_bb, pos_mask, stride)
                total_bbox[level] += bbox_l.item() if isinstance(bbox_l, torch.Tensor) else 0
                loss = loss + obj_l + bbox_l

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            total_batches += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_norm_sum += batch_grad_norm

        total_loss += loss.item()
        total_batches += 1

    n = max(total_batches, 1)
    return (total_loss / n,
            {k: v / n for k, v in total_obj.items()},
            {k: v / n for k, v in total_bbox.items()},
            grad_norm_sum / n)


@torch.no_grad()
def validate_v5(model, loader, criterion_cls, device):
    model.eval()
    val_obj = {"p2": 0.0, "p3": 0.0, "p4": 0.0}
    tp = {"p2": 0, "p3": 0, "p4": 0}
    fp = {"p2": 0, "p3": 0, "p4": 0}
    fn = {"p2": 0, "p3": 0, "p4": 0}
    n_batches = 0

    fpn_levels = ["p2", "p3", "p4"]

    for batch in tqdm(loader, desc="Val v5"):
        (patches, hm_p2, bb_p2, hm_p3, bb_p3, hm_p4, bb_p4) = batch
        patches = patches.to(device)
        targets = {
            "p2": (hm_p2.to(device), bb_p2.to(device)),
            "p3": (hm_p3.to(device), bb_p3.to(device)),
            "p4": (hm_p4.to(device), bb_p4.to(device)),
        }

        outputs = model(patches)
        for level in fpn_levels:
            pred_obj = outputs[f"{level}_obj"]
            t_hm, _ = targets[level]
            val_obj[level] += criterion_cls(pred_obj, t_hm).item()

            pred_probs = torch.sigmoid(pred_obj)
            gt_bin = (t_hm > 0.5).float()
            pred_bin = (pred_probs > 0.5).float()
            tp[level] += (pred_bin * gt_bin).sum().item()
            fp[level] += (pred_bin * (1 - gt_bin)).sum().item()
            fn[level] += ((1 - pred_bin) * gt_bin).sum().item()

        n_batches += 1

    n = max(n_batches, 1)
    results = {}
    for level in fpn_levels:
        p = tp[level] / max(tp[level] + fp[level], 1)
        r = tp[level] / max(tp[level] + fn[level], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[level] = {
            "obj_loss": val_obj[level] / n,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    return results


class WiderFaceFPNMineDataset(Dataset):
    def __init__(self, root_dir, split="train", target_h=480, target_w=640,
                 samples_per_image=4, max_crop_attempts=50):
        self.root_dir = root_dir
        self.target_h = target_h
        self.target_w = target_w
        self.samples_per_image = samples_per_image
        self.max_crop_attempts = max_crop_attempts
        self.samples = []

        img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        annot_file = os.path.join(root_dir, "wider_face_split",
                                  f"wider_face_{split}_bbx_gt.txt")
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")

        with open(annot_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if i >= len(lines) or not img_name:
                break
            if "/" not in img_name:
                continue
            num_faces = int(lines[i].strip())
            i += 1
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                for _ in range(num_faces):
                    i += 1
                continue
            faces = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    if w > 5 and h > 5:
                        faces.append((x, y, w, h))
            self.samples.append((img_path, faces))
            if num_faces == 0:
                self.samples.append((img_path, []))

        n_full = len(self.samples)
        n_eff = n_full * samples_per_image
        print(f"WiderFaceFPNMineDataset: {n_full} images, ~{n_eff} crops ({split})")

    def __len__(self):
        return len(self.samples) * self.samples_per_image

    def _crop_avoids_faces(self, h, w, crop_h, crop_w, faces):
        for _ in range(self.max_crop_attempts):
            x1 = np.random.randint(0, max(1, w - crop_w))
            y1 = np.random.randint(0, max(1, h - crop_h))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            ok = True
            for fx, fy, fw, fh in faces:
                ix1, iy1 = max(x1, fx), max(y1, fy)
                ix2, iy2 = min(x2, fx + fw), min(y2, fy + fh)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = (ix2 - ix1) * (iy2 - iy1)
                    face_area = fw * fh
                    if overlap > 0.3 * min(face_area, crop_w * crop_h):
                        ok = False
                        break
            if ok:
                return y1, y2, x1, x2
        return 0, min(crop_h, h), 0, min(crop_w, w)

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        img_path, faces = self.samples[img_idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.target_w, self.target_h))
        scale_x = self.target_w / ow
        scale_y = self.target_h / oh
        scaled_faces = [(int(fx * scale_x), int(fy * scale_y),
                         int(fw * scale_x), int(fh * scale_y))
                        for fx, fy, fw, fh in faces]
        y1, y2, x1, x2 = self._crop_avoids_faces(
            self.target_h, self.target_w, self.target_h, self.target_w, scaled_faces)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] != self.target_h or patch.shape[1] != self.target_w:
            patch = cv2.resize(patch, (self.target_w, self.target_h))
        img_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
        h, w = self.target_h, self.target_w
        return (
            img_tensor,
            torch.zeros((1, h // 2, w // 2)),
            torch.zeros((4, h // 2, w // 2)),
            torch.zeros((1, h // 4, w // 4)),
            torch.zeros((4, h // 4, w // 4)),
            torch.zeros((1, h // 8, w // 8)),
            torch.zeros((4, h // 8, w // 8)),
        )


def hard_negative_mine_v5(model, dataset, device, top_k=100):
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=2, pin_memory=True)
    candidates = []
    n_eval = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Mining v5"):
            patches = batch[0].to(device)
            outputs = model(patches)
            for b in range(patches.size(0)):
                n_eval += 1
                max_conf = 0.0
                for level in ["p2", "p3", "p4"]:
                    obj = torch.sigmoid(outputs[f"{level}_obj"][b, 0])
                    m = obj.max().item()
                    if m > max_conf:
                        max_conf = m
                if max_conf > 0.3:
                    candidates.append((max_conf, patches[b].cpu()))
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept = min(top_k, len(candidates))
    fp_rate = len(candidates) / max(n_eval, 1)
    print(f"  Mined {len(candidates)} hard negatives from {n_eval} crops "
          f"(FP rate={fp_rate:.4f}), keeping top-{kept}")
    return [c[1] for c in candidates[:kept]]


class AnchorFreeGIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_bbox, target_bbox, pos_mask, stride):
        pm = pos_mask > 0.5
        if pm.sum() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)

        B = pred_bbox.size(0)
        device = pred_bbox.device

        pred_dx = pred_bbox[:, 0]
        pred_dy = pred_bbox[:, 1]
        pred_dw = pred_bbox[:, 2]
        pred_dh = pred_bbox[:, 3]
        target_dx = target_bbox[:, 0]
        target_dy = target_bbox[:, 1]
        target_dw = target_bbox[:, 2]
        target_dh = target_bbox[:, 3]

        gh, gw = pred_bbox.size(-2), pred_bbox.size(-1)
        ys, xs = torch.meshgrid(
            torch.arange(gh, device=device, dtype=torch.float32),
            torch.arange(gw, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xs = xs.unsqueeze(0).expand(B, -1, -1)
        ys = ys.unsqueeze(0).expand(B, -1, -1)

        p_cx = (xs + 0.5 + pred_dx) * stride
        p_cy = (ys + 0.5 + pred_dy) * stride
        pred_dw = torch.clamp(pred_dw, max=5.0)
        pred_dh = torch.clamp(pred_dh, max=5.0)
        p_w = torch.clamp(torch.exp(pred_dw) * stride, min=2.0, max=10000)
        p_h = torch.clamp(torch.exp(pred_dh) * stride, min=2.0, max=10000)

        t_cx = (xs + 0.5 + target_dx) * stride
        t_cy = (ys + 0.5 + target_dy) * stride
        t_w = torch.exp(target_dw) * stride
        t_h = torch.exp(target_dh) * stride

        p_x1, p_y1 = p_cx - p_w / 2, p_cy - p_h / 2
        p_x2, p_y2 = p_cx + p_w / 2, p_cy + p_h / 2
        t_x1, t_y1 = t_cx - t_w / 2, t_cy - t_h / 2
        t_x2, t_y2 = t_cx + t_w / 2, t_cy + t_h / 2

        xi = torch.clamp(torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1), min=0)
        yi = torch.clamp(torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1), min=0)
        inter = xi * yi
        p_area = p_w * p_h
        t_area = t_w * t_h
        union = p_area + t_area - inter + 1e-8
        iou = inter / union

        c_x1 = torch.min(p_x1, t_x1)
        c_y1 = torch.min(p_y1, t_y1)
        c_x2 = torch.max(p_x2, t_x2)
        c_y2 = torch.max(p_y2, t_y2)
        c_area = torch.clamp(c_x2 - c_x1, min=0) * torch.clamp(c_y2 - c_y1, min=0)
        giou = iou - (c_area - union) / (c_area + 1e-8)

        pmf = pm.float().squeeze(1)
        giou = giou * pmf
        giou = torch.nan_to_num(giou, nan=0.0, posinf=0.0, neginf=0.0)
        loss_per = (1 - giou) * pmf
        n_pos = pmf.sum().clamp(min=1)
        return loss_per.sum() / n_pos


class AnchorFreeCIoULoss(nn.Module):
    """Complete IoU loss for anchor-free detection.
    
    Adds center distance + aspect ratio penalty to GIoU.
    CIoU = IoU - ρ²(b, b_gt)/c² - αv
    
    Where:
      ρ²: squared center distance
      c²: squared diagonal of smallest enclosing box
      v: aspect ratio consistency (inverse, more similar = lower penalty)
      α: trade-off, α = v / (1 - IoU + v)
    
    Key benefit over GIoU: non-zero gradient even for non-overlapping boxes
    (center distance term always provides gradient).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_bbox, target_bbox, pos_mask, stride):
        pm = pos_mask > 0.5
        if pm.sum() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)

        B = pred_bbox.size(0)
        device = pred_bbox.device

        pred_dx = pred_bbox[:, 0]
        pred_dy = pred_bbox[:, 1]
        pred_dw = pred_bbox[:, 2]
        pred_dh = pred_bbox[:, 3]
        target_dx = target_bbox[:, 0]
        target_dy = target_bbox[:, 1]
        target_dw = target_bbox[:, 2]
        target_dh = target_bbox[:, 3]

        gh, gw = pred_bbox.size(-2), pred_bbox.size(-1)
        ys, xs = torch.meshgrid(
            torch.arange(gh, device=device, dtype=torch.float32),
            torch.arange(gw, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xs = xs.unsqueeze(0).expand(B, -1, -1)
        ys = ys.unsqueeze(0).expand(B, -1, -1)

        p_cx = (xs + 0.5 + pred_dx) * stride
        p_cy = (ys + 0.5 + pred_dy) * stride
        pred_dw = torch.clamp(pred_dw, max=5.0)
        pred_dh = torch.clamp(pred_dh, max=5.0)
        p_w = torch.clamp(torch.exp(pred_dw) * stride, min=2.0, max=10000)
        p_h = torch.clamp(torch.exp(pred_dh) * stride, min=2.0, max=10000)

        t_cx = (xs + 0.5 + target_dx) * stride
        t_cy = (ys + 0.5 + target_dy) * stride
        t_w = torch.exp(target_dw) * stride
        t_h = torch.exp(target_dh) * stride

        p_x1, p_y1 = p_cx - p_w / 2, p_cy - p_h / 2
        p_x2, p_y2 = p_cx + p_w / 2, p_cy + p_h / 2
        t_x1, t_y1 = t_cx - t_w / 2, t_cy - t_h / 2
        t_x2, t_y2 = t_cx + t_w / 2, t_cy + t_h / 2

        # IoU computation
        xi = torch.clamp(torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1), min=0)
        yi = torch.clamp(torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1), min=0)
        inter = xi * yi
        p_area = p_w * p_h
        t_area = t_w * t_h
        union = p_area + t_area - inter + 1e-8
        iou = inter / union

        # Center distance ρ²(b, b_gt)
        rho2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

        # Diagonal of smallest enclosing box c²
        c_x1 = torch.min(p_x1, t_x1)
        c_y1 = torch.min(p_y1, t_y1)
        c_x2 = torch.max(p_x2, t_x2)
        c_y2 = torch.max(p_y2, t_y2)
        c2 = torch.clamp(c_x2 - c_x1, min=0) ** 2 + torch.clamp(c_y2 - c_y1, min=0) ** 2 + 1e-8

        # Aspect ratio consistency v
        p_ar = p_w / (p_h + 1e-8)
        t_ar = t_w / (t_h + 1e-8)
        v = (4 / (torch.pi ** 2)) * (torch.atan(p_ar) - torch.atan(t_ar)) ** 2

        # Trade-off α
        alpha = v / (1 - iou + v + 1e-8)

        # CIoU = IoU - ρ²/c² - αv
        ciou = iou - rho2 / c2 - alpha * v

        pmf = pm.float().squeeze(1)
        ciou = ciou * pmf
        ciou = torch.nan_to_num(ciou, nan=0.0, posinf=0.0, neginf=0.0)
        loss_per = (1 - ciou) * pmf
        n_pos = pmf.sum().clamp(min=1)
        return loss_per.sum() / n_pos


def hard_negative_mine(model, dataset, device, top_k=500, batch_size=128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)
    candidates = []
    n_evaluated = 0
    with torch.no_grad():
        for patches, _, _, _ in tqdm(loader, desc="Mining"):
            patches = patches.to(device)
            outputs = model(patches)
            probs = torch.sigmoid(outputs[:, 0])
            for b in range(patches.size(0)):
                n_evaluated += 1
                max_conf = probs[b].max().item()
                if max_conf > 0.3:
                    candidates.append((max_conf, patches[b].cpu()))
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept = min(top_k, len(candidates))
    fp_rate = len(candidates) / max(n_evaluated, 1)
    print(f"  Mined {len(candidates)} hard negatives from {n_evaluated} crops "
          f"(FP rate={fp_rate:.4f}), keeping top-{kept}")
    return [c[1] for c in candidates[:kept]]


def adamw_effective_lr(optimizer, model):
    stats = {}
    for group in optimizer.param_groups:
        group_lr = group["lr"]
        wd = group.get("weight_decay", 0)
        for p in group["params"]:
            if p not in optimizer.state:
                continue
            state = optimizer.state[p]
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                continue
            p_name = None
            for name, param in model.named_parameters():
                if param is p:
                    p_name = name
                    break
            if p_name is None:
                continue
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_norm = exp_avg.norm().item()
            exp_avg_sq_norm = exp_avg_sq.norm().item()
            denom = (exp_avg_sq.sqrt() + 1e-8).norm().item()
            effective_step = group_lr * exp_avg_norm / max(denom, 1e-10)
            stats[p_name] = {
                "effective_lr": effective_step,
                "exp_avg_norm": exp_avg_norm,
                "exp_avg_sq_norm": exp_avg_sq_norm,
                "weight_decay_step": group_lr * wd,
            }
    return stats


def weight_l1_movement(prev_weights, curr_weights):
    stats = {}
    for name in prev_weights:
        if name in curr_weights:
            prev = prev_weights[name]
            curr = curr_weights[name]
            l1 = np.abs(curr - prev).sum()
            prev_norm = np.abs(prev).sum()
            stats[name] = {
                "l1_dist": float(l1),
                "l1_ratio": float(l1 / max(prev_norm, 1e-10)),
            }
    if stats:
        stats["_mean_l1_ratio"] = float(np.mean([v["l1_ratio"] for v in stats.values()]))
    return stats


def gradient_histograms(model, device, n_bins=25):
    histograms = {}
    all_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grads = param.grad.cpu().numpy().flatten()
            if not np.all(np.isfinite(grads)):
                continue
            all_grads.append(grads)
            hist, edges = np.histogram(grads, bins=n_bins)
            hist = hist / max(hist.sum(), 1)
            histograms[name] = {
                "histogram": hist.tolist(),
                "bin_edges": edges.tolist(),
                "mean": float(grads.mean()),
                "std": float(grads.std()),
                "min": float(grads.min()),
                "max": float(grads.max()),
                "pct_zero": float((grads == 0).mean() * 100),
            }
    if all_grads:
        all_g = np.concatenate(all_grads)
        histograms["_global"] = {
            "mean": float(all_g.mean()),
            "std": float(all_g.std()),
            "min": float(all_g.min()),
            "max": float(all_g.max()),
            "pct_zero": float((all_g == 0).mean() * 100),
        }
    return histograms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/face/widerface")
    parser.add_argument("--output", default="models/face_cnn.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-cosine", action="store_true")
    parser.add_argument("--no-hardmine", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--log-csv", default="models/training_metrics.csv")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--min-f1", type=float, default=0.08)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--no-focal", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--no-giou", action="store_true")
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--multiscale-sizes", type=int, nargs="+",
                        default=[96, 112, 128, 144, 160, 176, 192])
    parser.add_argument("--no-depthwise", action="store_true",
                        help="Disable depthwise separable convolutions")
    parser.add_argument("--no-se", action="store_true",
                        help="Disable Squeeze-and-Excitation block")
    parser.add_argument("--num-anchors", type=int, default=3,
                        help="Number of anchor boxes per grid cell (default: 3)")
    parser.add_argument("--mosaic", action="store_true",
                        help="Enable mosaic augmentation (mixes 4 images)")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable mixup augmentation (blends pairs of samples)")
    parser.add_argument("--cutout", action="store_true",
                        help="Enable cutout augmentation (random erasing)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor for objectness loss (0=disabled, recommended)")
    parser.add_argument("--neg-pos-ratio", type=int, default=10,
                        help="Negative:positive sampling ratio for balanced obj loss (default: 10)")
    parser.add_argument("--bg-ratio", type=float, default=0.3,
                        help="Fraction of training crops that are pure background (default: 0.3)")
    parser.add_argument("--validate-interval", type=int, default=1,
                        help="Run validation every N epochs (default: 1, set to 7 to save time)")
    parser.add_argument("--arch", type=str, default="v4",
                        choices=["v4", "v5"],
                        help="FaceCNN architecture to train: v4 (crop-based, anchor) or v5 (full-frame, FPN)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # Architecture Selection: v4 (crop-based anchor) or v5 (full-frame FPN)
    # =========================================================================
    if args.arch == "v5":
        from src.cv.face_detector_cnn import FaceFCNv5

        print("=" * 60)
        print("  FaceFCN v5.0 — Full-Frame Anchor-Free FPN Architecture")
        print("  Training on full 640x480 images with per-FPN-level targets")
        print("=" * 60)

        v5_batch = max(1, min(args.batch_size, 16))
        if v5_batch != args.batch_size:
            print(f"  Note: v5 batch size capped at 16 (full-frame training), using {v5_batch}")

        train_dataset = WiderFaceFPNDataset(
            args.data, "train", target_h=480, target_w=640,
            augment=not args.no_augment)
        val_dataset = WiderFaceFPNDataset(
            args.data, "val", target_h=480, target_w=640,
            augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=v5_batch,
            shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=v5_batch,
            shuffle=False, num_workers=2, pin_memory=True)

        model = FaceFCNv5().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        print(f"Model: FaceFCNv5 — {total_params:,} params, {model_size_mb:.2f} MB")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        if args.no_cosine:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=1, eta_min=1e-5)

        criterion_cls = FocalLoss(gamma=args.focal_gamma,
                                  alpha=args.focal_alpha,
                                  reduction="balanced",
                                  neg_pos_ratio=args.neg_pos_ratio)
        criterion_bbox = AnchorFreeGIoULoss()

        use_amp = args.amp and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        ema = None
        if args.ema_decay > 0.0:
            ema = ModelEMA(model, decay=args.ema_decay)
            print(f"EMA enabled: decay={args.ema_decay}")

        total_train_imgs = len(train_dataset)
        n_batches_per_epoch = max(1, total_train_imgs // v5_batch)

        print(f"\n  Training samples: {total_train_imgs}")
        print(f"  Batch size: {v5_batch} | Batches/epoch: {n_batches_per_epoch}")
        print(f"  Epochs: {args.epochs} | Warmup: {args.warmup}")
        print(f"  LR: {args.lr} | Optimizer: AdamW | Scheduler: CosineAnnealing")
        print(f"  Loss: BalancedFocalLoss (γ={args.focal_gamma}, α={args.focal_alpha}) + AnchorFreeGIoU")
        print(f"  AMP: {use_amp} | EMA: {args.ema_decay}")
        print("=" * 60)

        csv_headers = [
            "epoch", "train_loss",
            "train_obj_p2", "train_obj_p3", "train_obj_p4",
            "train_bbox_p2", "train_bbox_p3", "train_bbox_p4",
            "val_obj_p2", "val_obj_p3", "val_obj_p4",
            "val_f1_p2", "val_f1_p3", "val_f1_p4",
            "lr", "epoch_time_s", "gpu_mem_mb", "grad_norm",
        ]
        metrics_path = args.log_csv.replace(".csv", "_v5.csv")

        start_epoch = 0
        best_val_f1 = 0.0
        cumulative_flops = 0.0
        prev_weight_dict = None
        metrics_rows = []

        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            best_val_f1 = ckpt.get("val_f1", 0.0)
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"Resumed from epoch {start_epoch}")

        ts_start = time.time()
        for epoch in range(args.epochs):
            epoch_t0 = time.time()
            if epoch < args.warmup:
                lr = args.lr * (epoch + 1) / args.warmup
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif not args.no_cosine:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_obj_dict, train_bbox_dict, grad_norm = train_epoch_v5(
                model, train_loader, optimizer, criterion_cls, criterion_bbox,
                device, scaler=scaler)

            if ema is not None:
                ema.update(model)

            if not args.no_hardmine and epoch > 0 and epoch % 10 == 0:
                print("  Hard-negative mining on full-frame background crops...")
                mine_dataset = WiderFaceFPNMineDataset(
                    args.data, "train", target_h=480, target_w=640,
                    samples_per_image=4)
                hard_negatives = hard_negative_mine_v5(
                    model, mine_dataset, device, top_k=100)
                if len(hard_negatives) >= 8:
                    neg_patches = torch.stack(hard_negatives[:8])
                    neg_hm = [torch.zeros((neg_patches.size(0), 1, h // s, w // s))
                              for s in (2, 4, 8)]
                    neg_bb = [torch.zeros((neg_patches.size(0), 4, h // s, w // s))
                              for s in (2, 4, 8)]
                    neg_ds = torch.utils.data.TensorDataset(
                        neg_patches, *neg_hm[0], *neg_bb[0],
                        *neg_hm[1], *neg_bb[1], *neg_hm[2], *neg_bb[2])
                    neg_loader = DataLoader(
                        neg_ds, batch_size=4, shuffle=True)
                    model.train()
                    for batch in neg_loader:
                        p = batch[0].to(device)
                        optimizer.zero_grad()
                        out = model(p)
                        loss = torch.tensor(0.0, device=device)
                        for li, level in enumerate(["p2", "p3", "p4"]):
                            loss = loss + criterion_cls(out[f"{level}_obj"], batch[1 + li*2].to(device))
                        if torch.isfinite(loss):
                            loss.backward()
                            optimizer.step()
                    print(f"  Fine-tuned on {len(hard_negatives)} hard negatives")

            do_val = (epoch % args.validate_interval == 0)
            if do_val:
                val_results = validate_v5(model, val_loader, criterion_cls, device)

            epoch_time = time.time() - epoch_t0
            gpu_mem = 0
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6
                torch.cuda.reset_peak_memory_stats()

            flops_est = 0.5e9 * n_batches_per_epoch
            cumulative_flops += flops_est
            epoch_gflops = flops_est / 1e9

            if do_val:
                avg_f1 = float(np.mean([v["f1"] for v in val_results.values()]))
                if avg_f1 > best_val_f1:
                    best_val_f1 = avg_f1
                    best_path = args.output.replace(".pth", "_best.pth")
                    checkpoint = {
                        'ema_state_dict': ema.state_dict() if ema is not None else model.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_f1': avg_f1,
                        'lr': current_lr,
                    }
                    torch.save(checkpoint, best_path)
                    if os.path.exists(args.output):
                        os.remove(args.output)
                    os.symlink(os.path.basename(best_path), args.output)

                row = [
                    epoch + 1, f"{train_loss:.4f}",
                    f"{train_obj_dict['p2']:.4f}", f"{train_obj_dict['p3']:.4f}", f"{train_obj_dict['p4']:.4f}",
                    f"{train_bbox_dict['p2']:.4f}", f"{train_bbox_dict['p3']:.4f}", f"{train_bbox_dict['p4']:.4f}",
                    f"{val_results['p2']['obj_loss']:.4f}", f"{val_results['p3']['obj_loss']:.4f}", f"{val_results['p4']['obj_loss']:.4f}",
                    f"{val_results['p2']['f1']:.4f}", f"{val_results['p3']['f1']:.4f}", f"{val_results['p4']['f1']:.4f}",
                    f"{current_lr:.8f}", f"{epoch_time:.2f}", f"{gpu_mem:.1f}", f"{grad_norm:.4f}",
                ]
                print(
                    f"Epoch {epoch+1:3d}/{args.epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Grad: {grad_norm:.2f} | "
                    f"F1: P2={val_results['p2']['f1']:.3f} P3={val_results['p3']['f1']:.3f} P4={val_results['p4']['f1']:.3f} | "
                    f"LR={current_lr:.1e} | "
                    f"t={epoch_time:.1f}s | mem={gpu_mem:.0f}MB" +
                    (" *" if avg_f1 == best_val_f1 else "")
                )
            else:
                row = [epoch + 1, f"{train_loss:.4f}"] + ["skip"] * 14 + [
                    f"{current_lr:.8f}", f"{epoch_time:.2f}", f"{gpu_mem:.1f}", f"{grad_norm:.4f}"]
                print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | Grad: {grad_norm:.2f} | LR={current_lr:.1e} | t={epoch_time:.1f}s")

            metrics_rows.append(row)

            epoch_ckpt = args.output.replace(".pth", f"_epoch_{epoch+1:02d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict() if ema is not None else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_f1": avg_f1 if do_val else 0.0,
                "grad_norm": grad_norm,
                "lr": current_lr,
            }, epoch_ckpt)

            with open(metrics_path, "w", newline="") as f:
                w_csv = csv.writer(f)
                w_csv.writerow(csv_headers)
                w_csv.writerows(metrics_rows)

            if do_val and epoch >= args.warmup and args.no_cosine:
                avg_obj = float(np.mean([v["obj_loss"] for v in val_results.values()]))
                scheduler.step(avg_obj)

        total_time = time.time() - ts_start
        print(f"\n{'='*60}")
        print(f"v5.0 Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best val F1: {best_val_f1:.4f}")
        print(f"Model: {args.output.replace('.pth', '_best.pth')} ({total_params:,} params)")
        return

    # =========================================================================
    # v4.0 Training: Crop-based anchor architecture (original pipeline)
    # =========================================================================
    use_depthwise = not args.no_depthwise
    use_se = not args.no_se
    use_multiscale = args.multiscale and not args.no_augment
    if args.multiscale and args.no_augment:
        print("Warning: --multiscale requires augmentation, disabling multiscale")

    train_dataset = WIDERFaceDataset(
        args.data, "train", args.input_size, augment=not args.no_augment,
        stride=8, sigma=args.sigma, num_anchors=args.num_anchors)
    train_dataset.is_multiscale = use_multiscale
    train_dataset.use_mosaic = args.mosaic and not args.no_augment
    train_dataset.use_mixup = args.mixup
    train_dataset.use_cutout = args.cutout and not args.no_augment
    train_dataset.bg_ratio = args.bg_ratio if not args.no_augment else 0.0

    val_dataset = WIDERFaceDataset(
        args.data, "val", args.input_size, augment=False,
        stride=8, sigma=args.sigma, num_anchors=args.num_anchors)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=collate_anchor)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=collate_anchor)

    model = FaceFCN(num_anchors=args.num_anchors,
                    use_depthwise=use_depthwise,
                    use_se=use_se).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size:   {model_size_mb:.2f} MB (float32)")
    print(f"Architecture: depthwise={use_depthwise}, SE={use_se}, anchors={args.num_anchors}")
    print(f"Augmentations: mosaic={args.mosaic}, mixup={args.mixup}, cutout={args.cutout}")
    print(f"Label smoothing: {args.label_smoothing}")

    flops_train = compute_fcn_flops(args.input_size, args.input_size,
                                    use_depthwise=use_depthwise, use_se=use_se)
    train_flops_per_sample = flops_train["total_flops"]
    train_flops_per_batch = train_flops_per_sample * args.batch_size
    train_flops_fwd_bwd = train_flops_per_batch * 3
    batches_per_epoch = len(train_dataset) // args.batch_size
    flops_per_epoch = train_flops_fwd_bwd * batches_per_epoch
    print(f"\n--- FLOPs Analysis ---")
    print(f"Training input:        {args.input_size}x{args.input_size}x3")
    print(f"Forward FLOPs/sample:  {train_flops_per_sample:,.0f} ({train_flops_per_sample/1e6:.1f} MFLOPs)")
    print(f"Fwd+Bwd FLOPs/batch:   {train_flops_fwd_bwd:,.0f} ({train_flops_fwd_bwd/1e9:.2f} GFLOPs)")
    print(f"Batches/epoch:         {batches_per_epoch}")
    print(f"FLOPs/epoch:           {flops_per_epoch:,.0f} ({flops_per_epoch/1e12:.3f} TFLOPs)")
    print(f"FLOPs total (50 ep):   {flops_per_epoch*50:,.0f} ({flops_per_epoch*50/1e12:.2f} TFLOPs)")

    flops_inf = compute_inference_flops(480, 640,
                                        use_depthwise=use_depthwise,
                                        use_se=use_se)
    print(f"\nInference (640x480, {flops_inf['n_scales']} scales, "
          f"depthwise={use_depthwise}, SE={use_se}):")
    for ps in flops_inf["per_scale"]:
        print(f"  scale={ps['scale']:.2f} -> {ps['w']}x{ps['h']}: {ps['gflops']:.2f} GFLOPs")
    print(f"  Total inference: {flops_inf['total_gflops']:.2f} GFLOPs")

    flops_old = compute_fcn_flops(args.input_size, args.input_size,
                                  use_depthwise=False, use_se=False)
    savings = (1 - flops_train["total_flops"] / flops_old["total_flops"]) * 100
    print(f"  FLOP reduction vs original: {savings:.1f}%")

    print(f"\nPer-layer FLOPs breakdown (training, 128x128):")
    for name, flops_c, desc in flops_train["layers"]:
        pct = flops_c / flops_train["total_flops"] * 100
        print(f"  {name:20s} {flops_c:>12,.0f} ({pct:5.1f}%){('  '+desc) if desc else ''}")
    print(f"  {'TOTAL':20s} {flops_train['total_flops']:>12,.0f} ({100.0:5.1f}%)")
    print(f"---")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.no_cosine:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.no_focal:
        criterion_cls = nn.BCEWithLogitsLoss(reduction="mean")
        print("Using BCEWithLogitsLoss (mean reduction)")
    else:
        criterion_cls = FocalLoss(gamma=args.focal_gamma,
                                  alpha=args.focal_alpha,
                                  reduction="balanced",
                                  neg_pos_ratio=args.neg_pos_ratio)
        print(f"Using BalancedFocalLoss (gamma={args.focal_gamma}, "
              f"alpha={args.focal_alpha}, neg:pos={args.neg_pos_ratio})")

    if args.no_giou:
        criterion_bbox = nn.SmoothL1Loss(reduction="sum")
    else:
        criterion_bbox = AnchorGIoULoss(stride=8,
                                        anchor_scales=[1.5, 3.0, 6.0][:args.num_anchors])

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    ema = None
    if args.ema_decay > 0.0:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"EMA enabled: decay={args.ema_decay}")

    convergence = ConvergenceDetector(
        window=10, eps_loss=0.001, eps_f1=0.005,
        min_epochs=12, min_f1=args.min_f1)

    start_epoch = 0
    best_val_obj = float("inf")
    best_val_f1 = 0.0
    cumulative_flops = 0.0
    prev_weight_dict = None
    metrics_rows = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_obj = ckpt.get("val_obj_loss", float("inf"))
        best_val_f1 = ckpt.get("val_f1", 0.0)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if os.path.exists(args.log_csv):
            import pandas as pd
            existing = pd.read_csv(args.log_csv)
            metrics_rows = existing.values.astype(str).tolist()
            cumulative_flops = float(existing["cumulative_tflops"].iloc[-1]) * 1e12 if "cumulative_tflops" in existing.columns else 0.0
            print(f"Resumed from epoch {start_epoch} ({len(metrics_rows)} prior metric rows)")
        for _ in range(start_epoch - 1):
            scheduler.step()
    else:
        if args.resume:
            print(f"Resume file not found: {args.resume}, starting from scratch")
        print("Training from scratch")

    deep_analysis_dir = args.output.replace(".pth", "_analysis")
    os.makedirs(deep_analysis_dir, exist_ok=True)

    csv_headers = [
        "epoch", "train_loss", "train_obj_loss", "train_bbox_loss",
        "train_pos_ratio", "val_obj_loss", "val_bbox_loss",
        "val_precision", "val_recall", "val_f1", "val_specificity", "val_pos_ratio",
        "val_mean_iou", "val_iou_at_05", "val_ece",
        "mean_dx_err", "mean_dy_err", "mean_ls_err",
        "lr", "epoch_time_s", "gpu_mem_mb", "grad_norm",
        "data_load_pct", "compute_pct",
        "max_update_ratio", "mean_weight_cosine_sim", "mean_l1_ratio",
        "dead_neuron_pct", "bn_mean_mean",
        "epoch_gflops", "cumulative_tflops", "loss_per_tflop",
    ]
    metrics_path = args.log_csv

    ts_start = time.time()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()

        if use_multiscale:
            chosen_size = np.random.choice(args.multiscale_sizes)
            _shared_train_size.value = chosen_size
            print(f"[Epoch {epoch+1}] Multi-scale size: {chosen_size}x{chosen_size}")
        else:
            _shared_train_size.value = args.input_size

        if epoch < args.warmup:
            lr = args.lr * (epoch + 1) / args.warmup
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        elif not args.no_cosine:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        (train_loss, train_obj, train_bbox, train_pos_ratio,
         load_time, compute_time, load_pct) = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_bbox, device,
            use_giou=not args.no_giou, scaler=scaler,
            use_mixup=args.mixup, num_anchors=args.num_anchors)

        if ema is not None:
            ema.update(model)

        grad_stats = layer_gradient_stats(model)
        grad_norm = grad_stats["_total_norm"]
        max_update_ratio = grad_stats["_max_update_ratio"]

        do_validate = (epoch % args.validate_interval == 0)
        if do_validate:
            v = validate(model, val_loader, criterion_cls, criterion_bbox, device,
                         use_giou=not args.no_giou, num_anchors=args.num_anchors)
        else:
            v = {k: 0.0 for k in ["val_obj_loss", "val_bbox_loss", "precision", "recall",
                  "f1", "specificity", "pos_ratio", "mean_iou", "iou_at_05", "ece",
                  "mean_dx_err", "mean_dy_err", "mean_ls_err"]}
        val_obj = v["val_obj_loss"]
        val_bbox = v["val_bbox_loss"]
        val_prec = v["precision"]
        val_rec = v["recall"]
        val_f1 = v["f1"]
        val_spec = v["specificity"]
        val_pos_ratio = v["pos_ratio"]
        val_mean_iou = v["mean_iou"]
        val_iou_at_05 = v["iou_at_05"]
        val_ece = v["ece"]
        mean_dx_err = v["mean_dx_err"]
        mean_dy_err = v["mean_dy_err"]
        mean_ls_err = v["mean_ls_err"]

        if args.no_cosine and epoch >= args.warmup:
            scheduler.step(val_obj)

        epoch_time = time.time() - epoch_t0
        gpu_mem = 0
        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats()

        epoch_gflops = flops_per_epoch / 1e9
        cumulative_flops += flops_per_epoch
        cumulative_tflops = cumulative_flops / 1e12
        loss_per_tflop = train_loss / max(epoch_gflops / 1000, 1e-12)

        w_stats = layer_weight_stats(model)
        dead_pct = w_stats["_total"]["near_zero_pct"]

        curr_state = model.state_dict()
        curr_weights = {k: v.cpu().numpy() for k, v in curr_state.items() if "weight" in k}
        cos_sims = {}
        l1_movement = {}
        if prev_weight_dict is not None:
            cos_sims = weight_cosine_similarity(prev_weight_dict, curr_weights)
            mean_wt_cos = cos_sims.get("_mean", 1.0)
            l1_movement = weight_l1_movement(prev_weight_dict, curr_weights)
            mean_l1_ratio = l1_movement.get("_mean_l1_ratio", 0.0)
        else:
            mean_wt_cos = 1.0
            mean_l1_ratio = 0.0
        prev_weight_dict = curr_weights

        eff_lr_stats = adamw_effective_lr(optimizer, model)

        epoch_analysis = {
            "epoch": epoch + 1,
            "weight_stats": {k: {sk: sv for sk, sv in v.items() if sk != "shape"}
                           for k, v in w_stats.items()},
            "gradient_stats": grad_stats,
            "weight_cosine_sim": cos_sims,
            "weight_l1_movement": l1_movement,
            "effective_lr": eff_lr_stats,
        }
        with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_analysis.json"), "w") as f:
            json.dump(epoch_analysis, f, indent=2, cls=NumpyEncoder)

        epoch_ckpt = args.output.replace(".pth", f"_epoch_{epoch+1:02d}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_obj_loss": val_obj,
            "val_f1": val_f1,
            "lr": current_lr,
        }, epoch_ckpt)

        is_best = False
        if do_validate and ((val_obj < best_val_obj - 1e-6) or (abs(val_obj - best_val_obj) < 1e-6 and val_f1 > best_val_f1)):
            is_best = True
            best_val_obj = val_obj
            best_val_f1 = val_f1
            best_path = args.output.replace(".pth", "_best.pth")
            save_dict = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(save_dict, best_path)
            if os.path.exists(args.output):
                os.remove(args.output)
            os.symlink(os.path.basename(best_path), args.output)

        bn_mean_mean = 0.0
        if do_validate and (epoch == 0 or epoch % 5 == 0):
            act_stats = activation_stats(model, train_loader, device)
            with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_activations.json"), "w") as f:
                json.dump(act_stats, f, indent=2, cls=NumpyEncoder)
            bn_s = bn_statistics(model)
            with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_bn.json"), "w") as f:
                json.dump(bn_s, f, indent=2, cls=NumpyEncoder)
            grad_hists = gradient_histograms(model, device)
            with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_gradients.json"), "w") as f:
                json.dump(grad_hists, f, indent=2, cls=NumpyEncoder)
            conf_bins = v.get("confidence_20_bins", {})
            if conf_bins:
                with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_confidence.json"), "w") as f:
                    json.dump(conf_bins, f, indent=2, cls=NumpyEncoder)

        row = [
            epoch + 1,
            f"{train_loss:.6f}", f"{train_obj:.6f}", f"{train_bbox:.6f}",
            f"{train_pos_ratio:.6f}",
            f"{val_obj:.6f}" if do_validate else "skip",
            f"{val_bbox:.6f}" if do_validate else "skip",
            f"{val_prec:.4f}" if do_validate else "skip",
            f"{val_rec:.4f}" if do_validate else "skip",
            f"{val_f1:.4f}" if do_validate else "skip",
            f"{val_spec:.4f}" if do_validate else "skip",
            f"{val_pos_ratio:.6f}" if do_validate else "skip",
            f"{val_mean_iou:.4f}" if do_validate else "skip",
            f"{val_iou_at_05:.4f}" if do_validate else "skip",
            f"{val_ece:.6f}" if do_validate else "skip",
            f"{mean_dx_err:.6f}" if do_validate else "skip",
            f"{mean_dy_err:.6f}" if do_validate else "skip",
            f"{mean_ls_err:.6f}" if do_validate else "skip",
            f"{current_lr:.8f}",
            f"{epoch_time:.2f}",
            f"{gpu_mem:.1f}",
            f"{grad_norm:.4f}",
            f"{load_pct:.1f}", f"{100-load_pct:.1f}",
            f"{max_update_ratio:.6f}", f"{mean_wt_cos:.6f}", f"{mean_l1_ratio:.6f}",
            f"{dead_pct:.2f}", f"{bn_mean_mean:.4f}",
            f"{epoch_gflops:.2f}",
            f"{cumulative_tflops:.4f}",
            f"{loss_per_tflop:.4f}",
        ]
        metrics_rows.append(row)

        with open(metrics_path, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(csv_headers)
            w_csv.writerows(metrics_rows)

        if do_validate:
            print(
                f"Epoch {epoch+1:2d}/{args.epochs} | "
                f"Train: {train_loss:.4f} (obj={train_obj:.4f}, bbox={train_bbox:.4f}) | "
                f"Val: obj={val_obj:.4f} F1={val_f1:.3f} IoU@.5={val_iou_at_05:.3f} | "
                f"LR={current_lr:.1e} | "
                f"t={epoch_time:.1f}s (load={load_pct:.0f}%) | "
                f"mem={gpu_mem:.0f}MB | "
                f"{chr(8711)}={max_update_ratio:.4f} | "
                f"dead={dead_pct:.1f}% | "
                f"cos={mean_wt_cos:.4f} | "
                f"ECE={val_ece:.4f}" +
                (" *" if is_best else "")
            )
        else:
            print(
                f"Epoch {epoch+1:2d}/{args.epochs} | "
                f"Train: {train_loss:.4f} (obj={train_obj:.4f}, bbox={train_bbox:.4f}) | "
                f"Val: skip | "
                f"LR={current_lr:.1e} | "
                f"t={epoch_time:.1f}s (load={load_pct:.0f}%) | "
                f"mem={gpu_mem:.0f}MB | "
                f"{chr(8711)}={max_update_ratio:.4f}"
            )

        if not args.no_hardmine and epoch % 3 == 0 and epoch > 0:
            print("  Hard-negative mining on full frames...")
            mine_dataset = FullFrameMineDataset(
                args.data, "train", input_size=args.input_size,
                samples_per_image=8, num_anchors=args.num_anchors)
            hard_negatives = hard_negative_mine(
                model, mine_dataset, device, top_k=200, batch_size=args.batch_size)
            if len(hard_negatives) >= args.batch_size:
                neg_patches = torch.stack(hard_negatives[:args.batch_size])
                neg_heatmaps = torch.zeros((neg_patches.size(0), 1, 16, 16))
                neg_bboxes = torch.zeros((neg_patches.size(0), 3, 16, 16))
                neg_anchors = torch.zeros((neg_patches.size(0), 16, 16), dtype=torch.long)
                neg_dataset = torch.utils.data.TensorDataset(
                    neg_patches, neg_heatmaps, neg_bboxes, neg_anchors)
                neg_loader = DataLoader(neg_dataset, batch_size=args.batch_size, shuffle=True)
                model.train()
                for neg_patch, neg_hm, neg_bb, neg_anc in neg_loader:
                    neg_patch, neg_hm = neg_patch.to(device), neg_hm.to(device)
                    neg_bb = neg_bb.to(device)
                    optimizer.zero_grad()
                    out = model(neg_patch)
                    loss = criterion_cls(out[:, 0:1], neg_hm)
                    loss.backward()
                    optimizer.step()
                print(f"  Fine-tuned on {len(hard_negatives)} hard negatives")
            else:
                print(f"  Only {len(hard_negatives)} hard negatives found, "
                      f"skipping fine-tune (need >= {args.batch_size})")

        if do_validate and convergence.update(epoch + 1, val_obj, val_f1, grad_norm, mean_wt_cos, mean_l1_ratio):
            print(f"Training converged at epoch {epoch+1} "
                  f"(val_loss={val_obj:.4f}, signals: {convergence.reason})")
            break

    total_time = time.time() - ts_start
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best val obj loss: {best_val_obj:.6f} | Best val F1: {best_val_f1:.4f}")
    print(f"Total FLOPs:       {cumulative_flops:,.0f} ({cumulative_tflops:.2f} TFLOPs)")
    print(f"Inference FLOPs:   {flops_inf['total_gflops']:.2f} GFLOPs (640x480, {flops_inf['n_scales']} scales)")
    print(f"Deep analysis dir: {deep_analysis_dir}/")
    print(f"Training log:      {metrics_path}")
    print(f"Model:             {args.output.replace('.pth', '_best.pth')} ({model_size_mb:.2f} MB, {total_params:,} params)")

    gpu_efficiency = 0.0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        theoretical_peak = 6.5
        measured_tflops = flops_per_epoch / 1e12 / max(total_time / max(epoch + 1, 1), 1e-10)
        gpu_efficiency = measured_tflops / theoretical_peak * 100
        print(f"GPU: {props.name} | Measured: {measured_tflops:.2f} TFLOPs/s ({gpu_efficiency:.1f}% of theoretical peak)")

    summary = {
        "total_time_s": total_time,
        "total_time_min": total_time / 60,
        "best_val_obj_loss": best_val_obj,
        "best_val_f1": best_val_f1,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "epochs_completed": epoch + 1,
        "total_tflops": round(cumulative_tflops, 4),
        "tflops_per_epoch": round(flops_per_epoch / 1e12, 6),
        "inference_gflops": round(flops_inf["total_gflops"], 4),
        "inference_scales": flops_inf["n_scales"],
        "inference_per_scale": flops_inf["per_scale"],
        "training_flops_per_sample": train_flops_per_sample,
        "training_flops_per_batch_fwd_bwd": train_flops_fwd_bwd,
        "flops_breakdown": [(n, f) for n, f, _ in flops_train["layers"]],
        "gpu_efficiency_pct": round(gpu_efficiency, 1) if device.type == "cuda" else None,
        "data_load_vs_compute_pct": {"load": load_pct, "compute": 100 - load_pct},
        "architecture": {
            "depthwise": use_depthwise,
            "se_block": use_se,
            "num_anchors": args.num_anchors,
            "anchor_scales": [1.5, 3.0, 6.0][:args.num_anchors],
        },
        "augmentations": {
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "cutout": args.cutout,
            "label_smoothing": args.label_smoothing,
        },
    }
    with open(args.output.replace(".pth", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"Summary saved to {args.output.replace('.pth', '_summary.json')}")


if __name__ == "__main__":
    main()
