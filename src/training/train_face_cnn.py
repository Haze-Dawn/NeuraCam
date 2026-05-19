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
    """Multi-signal convergence detector.
    Uses 6 signals + 1 guard to determine when training has genuinely plateaued.
    
    Signals:
    1. Val loss not improving (relative tolerance)
    2. Val F1 not improving (absolute tolerance)
    3. Gradient norm small/stable
    4. Weight cosine similarity near 1.0 (weights frozen)
    5. Weight L1 movement near zero
    6. Linear slope of val loss over window near zero
    
    Guard:
    7. Minimum F1 threshold — prevents stopping at a trivial local minimum
       where the model has learned to predict low confidence for everything.
    
    All 6 signals must indicate convergence AND F1 must be above threshold.
    """
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

        # 1. Val loss: best in window close to global best?
        best_in_window = min(h["val_loss"] for h in window)
        loss_improved = (self.best_val_loss - best_in_window) > self.eps_loss

        # 2. F1: best in window close to global best?
        best_f1_window = max(h["val_f1"] for h in window)
        f1_improved = (best_f1_window - self.best_f1) > self.eps_f1

        # 3. Gradient norm — small means model settled
        avg_grad = float(np.mean([h["grad_norm"] for h in window]))
        grad_small = avg_grad < 0.5

        # 4. Weight cosine sim near 1.0 → weights frozen
        avg_cos = float(np.mean([h["wt_cos"] for h in window]))
        weights_frozen = avg_cos > 0.9999

        # 5. Weight L1 ratio → near zero → no meaningful change
        avg_l1 = float(np.mean([h["l1_ratio"] for h in window]))
        l1_tiny = avg_l1 < 0.003

        # 6. Linear slope of val loss over window
        epochs_a = np.array([h["epoch"] for h in window], dtype=np.float64)
        loss_a = np.array([h["val_loss"] for h in window], dtype=np.float64)
        if len(window) >= 3:
            slope = np.polyfit(epochs_a, loss_a, 1)[0]
        else:
            slope = 0.0
        slope_flat = abs(slope) < self.eps_loss / 10

        # 7. Guard: F1 must be above minimum threshold
        #    Prevents convergence at a trivial minimum where recall has collapsed.
        current_f1 = val_f1
        f1_above_floor = current_f1 >= self.min_f1

        # Decision: no improvement AND (slope flat OR weights frozen OR grad small) AND L1 tiny
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
            if l1_tiny: signals.append(f"l1={avg_l1:.4f}")
            self.reason = ", ".join(signals)
        elif not f1_above_floor and not_improving:
            self.reason = f"F1={current_f1:.4f} below min_f1={self.min_f1} — continuing"

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

# Shared memory variable for multi-scale training.
# DataLoader workers read this value in __getitem__ to determine the
# target patch size for the current epoch. Set before iterating the loader.
_shared_train_size = Value("i", 128)  # int32, default 128


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class GIoULoss(nn.Module):
    def __init__(self, stride=8, grid_size=16):
        super().__init__()
        self.stride = stride
        self.grid_size = grid_size

    def forward(self, pred_bbox, target_bbox, pos_mask):
        B = pred_bbox.size(0)
        pm = pos_mask > 0.5
        if pm.sum() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)

        actual_grid_size = pred_bbox.size(-1)
        pred_dx = pred_bbox[:, 0]
        pred_dy = pred_bbox[:, 1]
        pred_ls = pred_bbox[:, 2]
        target_dx = target_bbox[:, 0]
        target_dy = target_bbox[:, 1]
        target_ls = target_bbox[:, 2]

        cell_coords = self._make_cell_coords(B, actual_grid_size).to(pred_bbox.device)
        cx = cell_coords[:, 0]
        cy = cell_coords[:, 1]

        p_cx = (cx + 0.5 + pred_dx) * self.stride
        p_cy = (cy + 0.5 + pred_dy) * self.stride
        p_s = torch.exp(pred_ls) * self.stride * 16
        t_cx = (cx + 0.5 + target_dx) * self.stride
        t_cy = (cy + 0.5 + target_dy) * self.stride
        t_s = torch.exp(target_ls) * self.stride * 16

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

        giou = giou * pm.float().squeeze(1)
        n_pos = pm.float().sum().clamp(min=1)
        giou_loss = (1 - giou).sum() / n_pos
        return giou_loss * 5.0

    def _make_cell_coords(self, B, grid_size=None):
        if grid_size is None:
            grid_size = self.grid_size
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

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def flops_conv2d(cin, cout, k, h, w):
    """MACs (multiply-accumulate) for Conv2d with stride=1, same padding."""
    return 2 * h * w * k * k * cin * cout

def flops_bn(c, h, w):
    """FLOPs for BatchNorm2d inference (affine transform)."""
    return 4 * h * w * c

def flops_relu(c, h, w):
    """FLOPs for ReLU (comparison)."""
    return 1 * h * w * c

def compute_fcn_flops(input_h=128, input_w=128):
    """Compute theoretical FLOPs for FaceFCN at a given input spatial size.
    Uses ceil(h/2) for MaxPool output to handle odd dimensions correctly.
    Returns dict with per-layer breakdown and totals.
    """
    layers = []
    total = 0.0
    h, w = input_h, input_w

    # Block 1: Conv(3→16, 5) → BN → ReLU → MP(2)
    c = flops_conv2d(3, 16, 5, h, w); total += c; layers.append(("block1_conv", c, f"{h}×{w}×3→16"))
    c = flops_bn(16, h, w); total += c; layers.append(("block1_bn", c, ""))
    c = flops_relu(16, h, w); total += c; layers.append(("block1_relu", c, ""))
    h, w = (h + 1) // 2, (w + 1) // 2  # ceil division for MaxPool

    # Block 2: Conv(16→32, 3) → BN → ReLU → MP(2)
    c = flops_conv2d(16, 32, 3, h, w); total += c; layers.append(("block2_conv", c, f"{h}×{w}×16→32"))
    c = flops_bn(32, h, w); total += c; layers.append(("block2_bn", c, ""))
    c = flops_relu(32, h, w); total += c; layers.append(("block2_relu", c, ""))
    h, w = (h + 1) // 2, (w + 1) // 2

    # Block 3: Conv(32→64, 3) → BN → ReLU → MP(2)
    c = flops_conv2d(32, 64, 3, h, w); total += c; layers.append(("block3_conv", c, f"{h}×{w}×32→64"))
    c = flops_bn(64, h, w); total += c; layers.append(("block3_bn", c, ""))
    c = flops_relu(64, h, w); total += c; layers.append(("block3_relu", c, ""))
    h, w = (h + 1) // 2, (w + 1) // 2

    # Block 4: Conv(64→128, 3, dilation=2) → BN → ReLU (no MP)
    c = flops_conv2d(64, 128, 3, h, w); total += c; layers.append(("block4_conv", c, f"{h}×{w}×64→128(d=2)"))
    c = flops_bn(128, h, w); total += c; layers.append(("block4_bn", c, ""))
    c = flops_relu(128, h, w); total += c; layers.append(("block4_relu", c, ""))
    # h, w unchanged

    # FPN Skip: Conv(64→128, 1) at block3 resolution (h, w before block4)
    c = flops_conv2d(64, 128, 1, h, w); total += c; layers.append(("skip_conv", c, f"{h}×{w}×64→128"))
    # FPN Fuse: Conv(256→128, 1) after concat
    c = flops_conv2d(256, 128, 1, h, w); total += c; layers.append(("fuse_conv", c, f"{h}×{w}×256→128"))

    # Head: Conv(128→4, 1)
    c = flops_conv2d(128, 4, 1, h, w); total += c; layers.append(("head_conv", c, f"{h}×{w}×128→4"))

    return {
        "layers": layers,
        "total_flops": total,
        "total_gflops": total / 1e9,
        "input_shape": (input_h, input_w),
    }

def compute_inference_flops(frame_h=480, frame_w=640, scales=None):
    """Compute total inference FLOPs for multi-scale FaceCNN detection.
    Returns dict with per-scale breakdown.
    Matches face_detector_cnn.py detect(): descending scales 1.0, 1/1.15, 1/1.15^2, ...
    """
    if scales is None:
        sf = 1.15
        scales = [1.0 / (sf ** i) for i in range(5)]
    per_scale = []
    total = 0.0
    for s in scales:
        h, w = int(frame_h * s), int(frame_w * s)
        f = compute_fcn_flops(h, w)
        per_scale.append({"scale": s, "h": h, "w": w, "gflops": round(f["total_gflops"], 4)})
        total += f["total_flops"]
    return {
        "per_scale": per_scale,
        "total_flops": total,
        "total_gflops": total / 1e9,
        "frame_shape": (frame_h, frame_w),
        "n_scales": len(scales),
    }


from src.cv.face_detector_cnn import FaceFCN


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


class WIDERFaceDataset(Dataset):
    def __init__(self, root_dir, split="train", input_size=128,
                 augment=False, stride=8, sigma=1.5):
        self.root_dir = root_dir
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
        self.augment = augment
        self.sigma = sigma
        self.samples = []
        self.is_multiscale = False

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

    def __getitem__(self, idx):
        img_path, faces = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        face = faces[np.random.randint(len(faces))]
        fx, fy, fw, fh = face
        cx = fx + fw / 2
        cy = fy + fh / 2
        face_size = max(fw, fh)

        # Multi-scale: use shared variable for target size during training.
        # The shared value is set per-epoch by the training loop.
        # DataLoader workers read the current value from shared memory.
        if self.augment and self.is_multiscale:
            target_size = _shared_train_size.value
        else:
            target_size = self.input_size
        grid_size = target_size // self.stride

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
        if self.augment:
            patch = apply_augmentation(patch)
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
        gx = new_cx * grid_size
        gy = new_cy * grid_size
        heatmap = gaussian_heatmap(grid_size, gy, gx, sigma=self.sigma)
        gx_int = int(round(gx))
        gy_int = int(round(gy))
        gx_int = np.clip(gx_int, 0, grid_size - 1)
        gy_int = np.clip(gy_int, 0, grid_size - 1)
        dx = gx - gx_int
        dy = gy - gy_int
        log_size = np.log(max(new_size * target_size / self.stride, 0.01))
        bbox_target = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        bbox_target[0, gy_int, gx_int] = dx
        bbox_target[1, gy_int, gx_int] = dy
        bbox_target[2, gy_int, gx_int] = log_size
        return (patch_tensor,
                torch.from_numpy(heatmap).unsqueeze(0),
                torch.from_numpy(bbox_target))


def train_epoch(model, loader, optimizer, criterion_cls, criterion_bbox,
                device, use_giou=False, scaler=None):
    model.train()
    total_loss = total_obj = total_bbox = 0.0
    total_pos_cells = total_cells = 0
    total_batches = 0
    load_time = 0.0
    compute_time = 0.0

    for patches, heatmaps, bboxes in tqdm(loader, desc="Train"):
        t0 = time.perf_counter()

        patches = patches.to(device)
        heatmaps = heatmaps.to(device)
        bboxes = bboxes.to(device)
        B = patches.size(0)

        t1 = time.perf_counter()
        load_time += t1 - t0

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(patches)
            pred_obj = outputs[:, 0:1]
            pred_bbox = outputs[:, 1:]

            obj_loss = criterion_cls(pred_obj, heatmaps)
            pos_mask = (heatmaps > 0.5).float()
            pos_count = pos_mask.sum().item()
            total_pos_cells += pos_count
            gs = heatmaps.size(-1)
            total_cells += B * gs * gs

            if pos_mask.sum() > 0:
                if use_giou:
                    bbox_loss = criterion_bbox(pred_bbox, bboxes, pos_mask)
                else:
                    bbox_loss = criterion_bbox(pred_bbox * pos_mask, bboxes * pos_mask)
                    bbox_loss = bbox_loss / (pos_mask.sum() + 1) * 5.0
            else:
                bbox_loss = torch.tensor(0.0, device=device)

            loss = obj_loss + bbox_loss

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
def validate(model, loader, criterion_cls, criterion_bbox, device, use_giou=False):
    """Full validation with objectness metrics, bbox quality, and calibration."""
    model.eval()
    val_obj_loss = 0.0
    val_bbox_loss = 0.0
    total_pos_cells = total_cells = 0
    tp = fp = fn = tn = 0
    n_batches = 0

    # Bbox error components (for positive cells only)
    total_bbox_cells = 0
    sum_dx_err = sum_dy_err = sum_logsize_err = 0.0

    # IoU tracking
    batch_ious = []
    iou_05_count = iou_05_total = 0

    # Calibration bins (10 bins for ECE, 20 bins for histogram)
    n_bins = 10
    bin_confidences = [0.0] * n_bins
    bin_accuracies = [0.0] * n_bins
    bin_counts = [0] * n_bins
    n_bins20 = 20
    bin20_counts = [0] * n_bins20
    bin20_confidences = [0.0] * n_bins20

    for patches, heatmaps, bboxes in tqdm(loader, desc="Val"):
        patches = patches.to(device)
        heatmaps = heatmaps.to(device)
        bboxes = bboxes.to(device)
        B = patches.size(0)

        outputs = model(patches)
        pred_obj = outputs[:, 0:1]
        pred_bbox = outputs[:, 1:]

        val_obj_loss += criterion_cls(pred_obj, heatmaps).item()
        pos_mask = (heatmaps > 0.5).float()
        pos_count = pos_mask.sum().item()
        total_pos_cells += pos_count
        gs = heatmaps.size(-1)
        total_cells += B * gs * gs

        if pos_mask.sum() > 0:
            if use_giou:
                bbox_l = criterion_bbox(pred_bbox, bboxes, pos_mask)
                val_bbox_loss += bbox_l.item()
            else:
                bbox_l = criterion_bbox(pred_bbox * pos_mask, bboxes * pos_mask)
                val_bbox_loss += (bbox_l / (pos_mask.sum() + 1) * 5.0).item()

        per_cell_ious = []
        for b in range(B):
            pm = pos_mask[b, 0]
            if pm.sum() == 0:
                continue
            cell_indices = torch.where(pm > 0.5)

            for cy, cx in zip(cell_indices[0], cell_indices[1]):
                p_dx = pred_bbox[b, 0, cy, cx].item()
                p_dy = pred_bbox[b, 1, cy, cx].item()
                p_ls = pred_bbox[b, 2, cy, cx].item()
                t_dx = bboxes[b, 0, cy, cx].item()
                t_dy = bboxes[b, 1, cy, cx].item()
                t_ls = bboxes[b, 2, cy, cx].item()

                sum_dx_err += abs(p_dx - t_dx)
                sum_dy_err += abs(p_dy - t_dy)
                sum_logsize_err += abs(p_ls - t_ls)
                total_bbox_cells += 1

                # Decode bboxes to approximate pixel space for IoU
                stride = 8
                p_cx = (cx + 0.5 + p_dx) * stride
                p_cy = (cy + 0.5 + p_dy) * stride
                p_s = np.exp(p_ls) * stride * 16
                t_cx = (cx + 0.5 + t_dx) * stride
                t_cy = (cy + 0.5 + t_dy) * stride
                t_s = np.exp(t_ls) * stride * 16

                # IoU between square boxes
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

        # Per-cell objectness metrics + calibration
        pred_probs = torch.sigmoid(pred_obj)
        pred_bin = (pred_probs > 0.5).float()
        gt_bin = (heatmaps > 0.5).float()
        tp += (pred_bin * gt_bin).sum().item()
        fp += (pred_bin * (1 - gt_bin)).sum().item()
        fn += ((1 - pred_bin) * gt_bin).sum().item()
        tn += ((1 - pred_bin) * (1 - gt_bin)).sum().item()

        # Calibration bins
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
                bin20_counts[bin20_idx] += 1
                bin20_confidences[bin20_idx] += cv

        n_batches += 1

    val_obj_loss /= n_batches
    val_bbox_loss /= n_batches
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)
    pos_ratio = total_pos_cells / max(total_cells, 1)

    # Bbox error components
    mean_dx_err = sum_dx_err / max(total_bbox_cells, 1)
    mean_dy_err = sum_dy_err / max(total_bbox_cells, 1)
    mean_ls_err = sum_logsize_err / max(total_bbox_cells, 1)
    mean_iou = np.mean(batch_ious) if batch_ious else 0.0
    iou_at_05 = iou_05_count / max(iou_05_total, 1)

    # Expected Calibration Error
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_conf = bin_confidences[i] / bin_counts[i]
            avg_acc = bin_accuracies[i] / bin_counts[i]
            ece += (bin_counts[i] / max(sum(bin_counts), 1)) * abs(avg_conf - avg_acc)

    # Confidence histogram (20 bins, normalized)
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
            "mean_confidence": float(np.mean([c / max(cnt, 1) for c, cnt in zip(bin20_confidences, bin20_counts)])),
        },
    }


def layer_weight_stats(model):
    """Compute per-layer weight statistics. Returns dict of layer_name → {mean, std, near_zero_pct, shape, spectral_norm, effective_rank}."""
    stats = {}
    total_weights = 0
    total_near_zero = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.data.cpu().numpy().flatten()
            w_mat = param.data.cpu().numpy().reshape(param.shape[0], -1)

            # SVD for spectral norm and effective rank
            try:
                sv = np.linalg.svd(w_mat, compute_uv=False)
                spectral_norm = float(sv[0])
                # Effective rank = entropy of normalized singular values
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
    """Compute per-layer gradient L2 norms and update-to-weight ratios."""
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
                "update_ratio": gn / max(wn, 1e-10),  # signal-to-noise
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
    """Collect activation statistics: mean activation, dead ReLU % per layer."""
    model.eval()
    # Register forward hooks to capture ReLU outputs
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

    # Run a single batch through
    patches, _, _ = next(iter(loader))
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
    """Collect BatchNorm running mean/var statistics across layers."""
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
    """Compute cosine similarity between consecutive epoch weight snapshots."""
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
    """Full-frame dataset for hard-negative mining.
    Loads full WIDER Face images and generates random 128x128 crops
    from regions that do NOT overlap with any face annotation.
    The model is evaluated on these crops to find false positive detections
    on background-only regions — exactly what it struggles with at inference.
    """
    def __init__(self, root_dir, split="train", input_size=128, stride=8,
                 samples_per_image=10, max_crop_attempts=50):
        self.root_dir = root_dir
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
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
        """Generate a random crop that does not overlap any face by >30%."""
        for _ in range(self.max_crop_attempts):
            x1 = np.random.randint(0, max(1, img_w - crop_size))
            y1 = np.random.randint(0, max(1, img_h - crop_size))
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            # Check overlap with all faces
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
        # Fallback: return a crop from the top-left corner (safe default)
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
        return patch_tensor, heatmap, bbox_target


def hard_negative_mine(model, dataset, device, top_k=500, batch_size=128):
    """Mine hard negatives from FULL IMAGE background regions.
    Uses FullFrameMineDataset to sample random 128x128 crops that
    contain NO face annotations. Any crop where the model predicts
    a confidence > 0.3 is a false positive — a hard negative.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)
    candidates = []
    n_evaluated = 0
    with torch.no_grad():
        for patches, heatmaps, _ in tqdm(loader, desc="Mining"):
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
    """Read AdamW state to compute effective learning rate per layer.
    Returns dict of param_name → {effective_lr, exp_avg_norm, exp_avg_sq_norm}.
    Cost: ~0ms (reads existing CPU tensors).
    """
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
            # Find the parameter name (reverse lookup from model)
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
    """L1 distance between consecutive epoch weights, normalized by weight magnitude.
    Returns dict of layer_name → {l1_dist, l1_ratio}.
    Cost: ~1ms (numpy on CPU).
    """
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
    """Build 25-bin histograms of gradient values per conv layer.
    Cost: ~10ms (GPU→CPU transfer of ~140K values + numpy histogram).
    """
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
    parser.add_argument("--no-cosine", action="store_true", help="Use ReduceLROnPlateau instead of CosineAnnealing")
    parser.add_argument("--no-hardmine", action="store_true", help="Disable hard-negative mining")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--log-csv", default="models/training_metrics.csv")
    parser.add_argument("--resume", default=None, help="Path to epoch checkpoint (e.g. models/face_cnn_epoch_32.pth)")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian heatmap sigma for objectness targets (default: 1.5)")
    parser.add_argument("--min-f1", type=float, default=0.08,
                        help="Minimum validation F1 before convergence detector fires (default: 0.08)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (default: 2.0)")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal loss alpha parameter (default: 0.25)")
    parser.add_argument("--no-focal", action="store_true",
                        help="Use standard BCEWithLogitsLoss instead of FocalLoss")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (torch.cuda.amp)")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay rate for weight averaging (0.0 = disabled, typical 0.999)")
    parser.add_argument("--no-giou", action="store_true",
                        help="Use SmoothL1Loss instead of GIoULoss for bbox regression")
    parser.add_argument("--multiscale", action="store_true",
                        help="Enable multi-scale training: random patch sizes [96-192] per epoch. "
                             "Makes model scale-invariant by training on variable input resolutions. "
                             "Requires augment to be enabled.")
    parser.add_argument("--multiscale-sizes", type=int, nargs="+",
                        default=[96, 112, 128, 144, 160, 176, 192],
                        help="List of target sizes for multi-scale training (default: 96 112 128 144 160 176 192)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    use_multiscale = args.multiscale and not args.no_augment
    if args.multiscale and args.no_augment:
        print("Warning: --multiscale requires augmentation, disabling multiscale")

    train_dataset = WIDERFaceDataset(
        args.data, "train", args.input_size, augment=not args.no_augment,
        stride=8, sigma=args.sigma)
    train_dataset.is_multiscale = use_multiscale
    val_dataset = WIDERFaceDataset(
        args.data, "val", args.input_size, augment=False,
        stride=8, sigma=args.sigma)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    model = FaceFCN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size:   {model_size_mb:.2f} MB (float32)")

    # --- FLOPs analysis ---
    flops_train = compute_fcn_flops(args.input_size, args.input_size)
    train_flops_per_sample = flops_train["total_flops"]
    train_flops_per_batch = train_flops_per_sample * args.batch_size
    train_flops_fwd_bwd = train_flops_per_batch * 3  # ~2x backward + 1x forward
    batches_per_epoch = len(train_dataset) // args.batch_size
    flops_per_epoch = train_flops_fwd_bwd * batches_per_epoch
    print(f"\n--- FLOPs Analysis ---")
    print(f"Training input:        {args.input_size}×{args.input_size}×3")
    print(f"Forward FLOPs/sample:  {train_flops_per_sample:,.0f} ({train_flops_per_sample/1e6:.1f} MFLOPs)")
    print(f"Fwd+Bwd FLOPs/batch:   {train_flops_fwd_bwd:,.0f} ({train_flops_fwd_bwd/1e9:.2f} GFLOPs)")
    print(f"Batches/epoch:         {batches_per_epoch}")
    print(f"FLOPs/epoch:           {flops_per_epoch:,.0f} ({flops_per_epoch/1e12:.3f} TFLOPs)")
    print(f"FLOPs total (50 ep):   {flops_per_epoch*50:,.0f} ({flops_per_epoch*50/1e12:.2f} TFLOPs)")

    # Inference FLOPs at processing resolution (640×480)
    flops_inf = compute_inference_flops(480, 640)
    print(f"\nInference (640×480, {flops_inf['n_scales']} scales):")
    for ps in flops_inf["per_scale"]:
        print(f"  scale={ps['scale']:.2f} → {ps['w']}×{ps['h']}: {ps['gflops']:.2f} GFLOPs")
    print(f"  Total inference: {flops_inf['total_gflops']:.2f} GFLOPs")

    # Per-layer breakdown
    print(f"\nPer-layer FLOPs breakdown (training, 128×128):")
    for name, flops, desc in flops_train["layers"]:
        pct = flops / flops_train["total_flops"] * 100
        label = f"  {name:20s}" + (f"  {desc}" if desc else "")
        print(f"  {name:20s} {flops:>12,.0f} ({pct:5.1f}%){('  '+desc) if desc else ''}")
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
    else:
        criterion_cls = FocalLoss(gamma=args.focal_gamma,
                                  alpha=args.focal_alpha,
                                   reduction="mean")

    if args.no_giou:
        criterion_bbox = nn.SmoothL1Loss(reduction="sum")
    else:
        criterion_bbox = GIoULoss(stride=8, grid_size=16)

    # AMP GradScaler
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # EMA (Exponential Moving Average) for inference weights
    ema = None
    if args.ema_decay > 0.0:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"EMA enabled: decay={args.ema_decay}")

    # --- Convergence detector ---
    convergence = ConvergenceDetector(
        window=10, eps_loss=0.001, eps_f1=0.005,
        min_epochs=12, min_f1=args.min_f1)

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_val_obj = float("inf")
    best_val_f1 = 0.0
    cumulative_flops = 0.0
    prev_weight_dict = None
    metrics_rows = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]  # 1-indexed, epochs 1..start_epoch already done
        best_val_obj = ckpt.get("val_obj_loss", float("inf"))
        best_val_f1 = ckpt.get("val_f1", 0.0)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # Load existing metrics CSV
        if os.path.exists(args.log_csv):
            import pandas as pd
            existing = pd.read_csv(args.log_csv)
            metrics_rows = existing.values.astype(str).tolist()
            cumulative_flops = float(existing["cumulative_tflops"].iloc[-1]) * 1e12 if "cumulative_tflops" in existing.columns else 0.0
            print(f"Resumed from epoch {start_epoch} ({len(metrics_rows)} prior metric rows)")
        # Advance scheduler to match resumed epoch
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

        # Multi-scale: pick a random training size for this epoch.
        # All batches in this epoch use the same size. DataLoader workers
        # read the size from the shared memory Value set here.
        if use_multiscale:
            chosen_size = np.random.choice(args.multiscale_sizes)
            _shared_train_size.value = chosen_size
            print(f"[Epoch {epoch+1}] Multi-scale size: {chosen_size}×{chosen_size}")
        else:
            _shared_train_size.value = args.input_size

        if epoch < args.warmup:
            lr = args.lr * (epoch + 1) / args.warmup
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        elif not args.no_cosine:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train ---
        (train_loss, train_obj, train_bbox, train_pos_ratio,
         load_time, compute_time, load_pct) = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_bbox, device,
            use_giou=not args.no_giou, scaler=scaler)

        # --- EMA update (after optimizer step, before validation) ---
        if ema is not None:
            ema.update(model)

        # --- Gradient analysis ---
        grad_stats = layer_gradient_stats(model)
        grad_norm = grad_stats["_total_norm"]
        max_update_ratio = grad_stats["_max_update_ratio"]

        # --- Validate (full metrics: loss, P/R/F1, IoU, bbox error, calibration) ---
        v = validate(model, val_loader, criterion_cls, criterion_bbox, device,
                      use_giou=not args.no_giou)
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

        # ReduceLROnPlateau needs val_loss — step after validation
        if args.no_cosine and epoch >= args.warmup:
            scheduler.step(val_obj)

        epoch_time = time.time() - epoch_t0

        # GPU memory
        gpu_mem = 0
        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats()

        # --- FLOPs tracking ---
        epoch_gflops = flops_per_epoch / 1e9
        cumulative_flops += flops_per_epoch
        cumulative_tflops = cumulative_flops / 1e12
        loss_per_tflop = train_loss / max(epoch_gflops / 1000, 1e-12)

        # --- Weight statistics (per-layer SVD, norms) ---
        w_stats = layer_weight_stats(model)
        dead_pct = w_stats["_total"]["near_zero_pct"]

        # --- Weight cosine similarity + L1 movement (epoch-over-epoch) ---
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

        # --- AdamW effective learning rate per layer (reads optimizer state, free) ---
        eff_lr_stats = adamw_effective_lr(optimizer, model)

        # --- Save weight/gradient/L1/eff-LR analysis every epoch ---
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

        # --- Epoch checkpoint ---
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

        # --- Best model (by val obj loss, tiebreak by F1) ---
        is_best = False
        if (val_obj < best_val_obj - 1e-6) or (abs(val_obj - best_val_obj) < 1e-6 and val_f1 > best_val_f1):
            is_best = True
            best_val_obj = val_obj
            best_val_f1 = val_f1
            best_path = args.output.replace(".pth", "_best.pth")
            save_dict = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(save_dict, best_path)
            if os.path.exists(args.output):
                os.remove(args.output)
            os.symlink(os.path.basename(best_path), args.output)

        # --- Periodic deep analysis (every 5 epochs) ---
        bn_mean_mean = 0.0
        if epoch == 0 or epoch % 5 == 0:
            # Activation statistics (single batch forward + CPU numpy, ~100ms)
            act_stats = activation_stats(model, train_loader, device)
            with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_activations.json"), "w") as f:
                json.dump(act_stats, f, indent=2, cls=NumpyEncoder)

            # BN statistics (reads stored running_mean/var, no forward pass, ~0ms)
            bn_s = bn_statistics(model)
            with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_bn.json"), "w") as f:
                json.dump(bn_s, f, indent=2, cls=NumpyEncoder)

            # Gradient histograms (GPU→CPU transfer + numpy, ~10ms)
            grad_hists = gradient_histograms(model, device)
            with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_gradients.json"), "w") as f:
                json.dump(grad_hists, f, indent=2, cls=NumpyEncoder)

            # Confidence histogram (20-bin, from validate's already-computed predictions, ~0ms)
            conf_bins = v.get("confidence_20_bins", {})
            if conf_bins:
                with open(os.path.join(deep_analysis_dir, f"epoch_{epoch+1:02d}_confidence.json"), "w") as f:
                    json.dump(conf_bins, f, indent=2, cls=NumpyEncoder)

        # --- Log line ---
        row = [
            epoch + 1,
            f"{train_loss:.6f}", f"{train_obj:.6f}", f"{train_bbox:.6f}",
            f"{train_pos_ratio:.6f}",
            f"{val_obj:.6f}", f"{val_bbox:.6f}",
            f"{val_prec:.4f}", f"{val_rec:.4f}", f"{val_f1:.4f}", f"{val_spec:.4f}", f"{val_pos_ratio:.6f}",
            f"{val_mean_iou:.4f}", f"{val_iou_at_05:.4f}", f"{val_ece:.6f}",
            f"{mean_dx_err:.6f}", f"{mean_dy_err:.6f}", f"{mean_ls_err:.6f}",
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
            w = csv.writer(f)
            w.writerow(csv_headers)
            w.writerows(metrics_rows)

        # --- Print summary ---
        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train: {train_loss:.4f} (obj={train_obj:.4f}, bbox={train_bbox:.4f}) | "
            f"Val: obj={val_obj:.4f} F1={val_f1:.3f} IoU@.5={val_iou_at_05:.3f} | "
            f"LR={current_lr:.1e} | "
            f"t={epoch_time:.1f}s (load={load_pct:.0f}%) | "
            f"mem={gpu_mem:.0f}MB | "
            f"∇={max_update_ratio:.4f} | "
            f"dead={dead_pct:.1f}% | "
            f"cos={mean_wt_cos:.4f} | "
            f"ECE={val_ece:.4f}" +
            (" ★" if is_best else "")
        )

        # --- Hard-negative mining on full-frame background (every 3rd epoch) ---
        if not args.no_hardmine and epoch % 3 == 0 and epoch > 0:
            print("  Hard-negative mining on full frames...")
            mine_dataset = FullFrameMineDataset(
                args.data, "train", input_size=args.input_size,
                samples_per_image=8)
            hard_negatives = hard_negative_mine(
                model, mine_dataset, device, top_k=200, batch_size=args.batch_size)
            if len(hard_negatives) >= args.batch_size:
                neg_patches = torch.stack(hard_negatives[:args.batch_size])
                neg_heatmaps = torch.zeros((neg_patches.size(0), 1, 16, 16))
                neg_bboxes = torch.zeros((neg_patches.size(0), 3, 16, 16))
                neg_dataset = torch.utils.data.TensorDataset(neg_patches, neg_heatmaps, neg_bboxes)
                neg_loader = DataLoader(neg_dataset, batch_size=args.batch_size, shuffle=True)
                model.train()
                for neg_patch, neg_hm, neg_bb in neg_loader:
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

        # --- Convergence check (multi-signal) ---
        if convergence.update(epoch + 1, val_obj, val_f1, grad_norm, mean_wt_cos, mean_l1_ratio):
            print(f"Training converged at epoch {epoch+1} "
                  f"(val_loss={val_obj:.4f}, signals: {convergence.reason})")
            break

    total_time = time.time() - ts_start
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best val obj loss: {best_val_obj:.6f} | Best val F1: {best_val_f1:.4f}")
    print(f"Total FLOPs:       {cumulative_flops:,.0f} ({cumulative_tflops:.2f} TFLOPs)")
    print(f"Inference FLOPs:   {flops_inf['total_gflops']:.2f} GFLOPs (640×480, {flops_inf['n_scales']} scales)")
    print(f"Deep analysis dir: {deep_analysis_dir}/")
    print(f"Training log:      {metrics_path}")
    print(f"Model:             {args.output.replace('.pth', '_best.pth')} ({model_size_mb:.2f} MB, {total_params:,} params)")

    gpu_efficiency = 0.0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        theoretical_peak = 6.5  # RTX 2060 theoretical peak TFLOPS FP32
        measured_tflops = flops_per_epoch / 1e12 / max(total_time / max(epoch + 1, 1), 1e-10)
        gpu_efficiency = measured_tflops / theoretical_peak * 100
        print(f"GPU: {props.name} | Measured: {measured_tflops:.2f} TFLOPs/s ({gpu_efficiency:.1f}% of theoretical peak)")

    # Save final summary
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
    }
    with open(args.output.replace(".pth", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"Summary saved to {args.output.replace('.pth', '_summary.json')}")


if __name__ == "__main__":
    main()
