"""
FaceCNN V0 — Full training pipeline for WIDER Face.

Trains the FaceCNN V0 architecture (186K params) from scratch
on WIDER Face with FocalLoss + EIoU loss, cosine annealing with
warmup, gradient clipping, ModelEMA, and periodic mAP validation.

Augmentations: RandomSquareCrop, horizontal flip, HSV jitter.
Target assignment: radius-based multi-positive (center_radius=2.5).

Usage:
    python src/training/train_face_cnn_v0.py --gpu --epochs 350 --batch 8

Output:
    models/face_cnn_v0/face_cnn_v0_best.pth
    models/face_cnn_v0/face_cnn_v0_final.pth
    models/face_cnn_v0/face_cnn_v0.onnx
    models/face_cnn_v0/training_metrics_v0.csv
"""
import os, sys, math, time, csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.training.architectures.face_cnn_v0 import FaceCNNV0

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(REPO, "models", "face_cnn_v0")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Loss Functions ──

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pos = (target > 0).float()
        neg = (target == 0).float()
        pos_loss = -pos * ((1 - pred) ** self.gamma) * pred.clamp(1e-8).log()
        neg_loss = -neg * (pred ** self.gamma) * (1 - pred).clamp(1e-8).log()
        pos_count = pos.sum()
        neg_count = neg.sum()
        if pos_count > 0 and neg_count > 0:
            neg_loss = neg_loss * (pos_count / neg_count).clamp(max=self.alpha)
        total = pos_loss.sum() + neg_loss.sum()
        denom = pos_count + neg_count * 0.1
        return total / denom.clamp(min=1)


class EIoULoss(nn.Module):
    """EIoU loss on decoded boxes. Operates on raw dx/dy/dw/dh offsets
    and decodes them internally. Computes IoU + center distance + wh penalty."""
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
        xs = xs.unsqueeze(0).expand(B, -1, -1)
        ys = ys.unsqueeze(0).expand(B, -1, -1)
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


# ── Metrics ──

def compute_map(predictions, targets, iou_thresh=0.5):
    if not predictions or not targets:
        return 0.0
    preds = sorted(predictions, key=lambda x: x[0], reverse=True)
    tp, fp = 0, 0
    matched = set()
    for score, box in preds:
        best_iou = 0
        best_idx = -1
        for ti, tbox in enumerate(targets):
            if ti in matched:
                continue
            iou = box_iou(box, tbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = ti
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    n_gt = len(targets)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (n_gt + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def box_iou(a, b):
    xo = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
    yo = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    inter = xo * yo
    return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-8)


# ── Dataset ──

class WIDERFaceDataset(Dataset):
    """WIDER Face with RandomSquareCrop and augmentations."""
    def __init__(self, split="train"):
        self.split = split
        self.crop_choices = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        self.img_dir = os.path.join(REPO, f"data/face/widerface/WIDER_{split}/images")
        label_path = os.path.join(REPO, f"data/face/widerface/wider_face_split/wider_face_{split}_bbx_gt.txt")
        self.samples = self._parse_labels(label_path)

    def _parse_labels(self, path):
        samples = []
        with open(path) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or '.jpg' not in line:
                continue
            img_name = line
            if i >= len(lines):
                break
            face_count_line = lines[i].strip()
            i += 1
            while face_count_line and not face_count_line.lstrip('-').replace(' ', '').replace('.' ,'').replace('e', '').lstrip('-').isdigit() and '.jpg' not in face_count_line:
                if i >= len(lines):
                    break
                face_count_line = lines[i].strip()
                i += 1
            if '.jpg' in face_count_line:
                img_name = face_count_line
                continue
            if not face_count_line:
                continue
            try:
                n_faces = int(face_count_line.split()[0])
            except (ValueError, IndexError):
                continue
            faces = []
            for _ in range(n_faces):
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines):
                    break
                parts = list(map(float, lines[i].strip().split()[:4]))
                faces.append(parts)
                i += 1
            samples.append((img_name, faces))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, faces = self.samples[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        if self.split == "train":
            # RandomSquareCrop: pick a crop factor, crop square, resize to 640
            crop_factor = np.random.choice(self.crop_choices)
            crop_size = min(h, w) * crop_factor
            crop_size = max(crop_size, 64)
            crop_size = min(crop_size, min(h, w))

            # Random crop position
            x_start = np.random.randint(0, max(1, int(w - crop_size)))
            y_start = np.random.randint(0, max(1, int(h - crop_size)))
            crop = img[y_start:y_start+int(crop_size), x_start:x_start+int(crop_size)]

            # Resize to 640x640
            isz = 640
            img = cv2.resize(crop, (isz, isz))

            # Adjust face annotations
            scale = isz / crop_size
            offset_x = x_start
            offset_y = y_start
        else:
            # Validation: resize keeping aspect ratio, pad to 640
            isz = 640
            scale = isz / max(h, w)
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh))
            canvas = np.zeros((isz, isz, 3), dtype=np.uint8)
            canvas[:nh, :nw] = img
            img = canvas
            offset_x = 0
            offset_y = 0

        # Augmentations (train only)
        if self.split == "train":
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            img = self._color_jitter(img)

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Build ground truth boxes (adjusted for crop)
        gt_boxes = []
        for f in faces:
            x1, y1, bw, bh = f[:4]
            if bw < 3 or bh < 3:
                continue
            # Adjust for crop offset
            x1 -= offset_x
            y1 -= offset_y
            if x1 + bw <= 0 or y1 + bh <= 0:
                continue
            # Clamp to crop bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x1 *= scale
            y1 *= scale
            bw *= scale
            bh *= scale
            cx = x1 + bw / 2
            cy = y1 + bh / 2
            if cx < 0 or cy < 0 or cx > isz or cy > isz:
                continue
            gt_boxes.append([cx, cy, bw, bh])

        if len(gt_boxes) == 0:
            gt_boxes = [[0, 0, 0, 0]]

        gt = torch.tensor(gt_boxes, dtype=torch.float32)
        return img_t, gt

    @staticmethod
    def _color_jitter(img):
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img


def collate_fn(batch):
    imgs, gts = zip(*batch)
    max_n = max(g.shape[0] for g in gts)
    padded = torch.zeros(len(gts), max_n, 4)
    for i, g in enumerate(gts):
        padded[i, :g.shape[0]] = g
    return torch.stack(imgs, 0), padded, torch.tensor([g.shape[0] for g in gts])


# ── Target Builder ──

def build_targets(gt_boxes, n_gts, feat_hw, strides, radius_factor=1.0):
    """Multi-positive radius-based assignment (center_radius ~2.5)."""
    targets = {}
    for lname, stride in [('8', 8), ('16', 16), ('32', 32)]:
        fh, fw = feat_hw[lname]
        targets[lname] = (
            torch.zeros(1, 1, fh, fw),
            torch.zeros(1, 1, fh, fw),
            torch.zeros(1, 4, fh, fw),
        )

    bs = gt_boxes.shape[0]
    for b in range(bs):
        n = int(n_gts[b])
        for i in range(n):
            cx, cy, bw, bh = gt_boxes[b, i].tolist()
            if bw < 3 or bh < 3:
                continue
            face_size = math.sqrt(bw * bh)
            target_sizes = np.array([32, 96, 256])
            stride_idx = np.argmin(np.abs(face_size - target_sizes))
            lname, stride = [('8', 8), ('16', 16), ('32', 32)][stride_idx]
            fh, fw = feat_hw[lname]

            gx = cx / stride
            gy = cy / stride

            radius = max(1.0, (face_size / stride) * radius_factor)
            radius = min(radius, 3.0)

            i_c = int(gx)
            j_c = int(gy)
            if i_c < 0 or i_c >= fw or j_c < 0 or j_c >= fh:
                continue

            r_int = int(math.ceil(radius))
            cls_t, obj_t, bbox_t = targets[lname]
            min_dist = float('inf')
            best_ij = (j_c, i_c)

            for dj in range(-r_int, r_int + 1):
                for di in range(-r_int, r_int + 1):
                    gj = j_c + dj
                    gi = i_c + di
                    if gi < 0 or gi >= fw or gj < 0 or gj >= fh:
                        continue
                    dist = math.sqrt((gi - gx) ** 2 + (gj - gy) ** 2)
                    if dist > radius:
                        continue
                    cls_t[0, 0, gj, gi] = 1.0
                    obj_t[0, 0, gj, gi] = 1.0
                    if dist < min_dist:
                        min_dist = dist
                        best_ij = (gj, gi)

            gj, gi = best_ij
            bbox_t[0, 0, gj, gi] = gx - gi - 0.5
            bbox_t[0, 1, gj, gi] = gy - gj - 0.5
            bbox_t[0, 2, gj, gi] = math.log(max(bw / stride, 1e-8))
            bbox_t[0, 3, gj, gi] = math.log(max(bh / stride, 1e-8))
    return targets


# ── Model EMA ──

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.lerp_(p, 1 - self.decay)
            self.updates += 1


# ── Validation ──

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_map = 0.0
    n = 0
    for imgs, gts, n_gts in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        for b in range(imgs.shape[0]):
            n_gt = int(n_gts[b])
            gt_boxes = gts[b, :n_gt].tolist()
            all_dets = []
            for lname, stride in [('8', 8), ('16', 16), ('32', 32)]:
                cls_p = torch.sigmoid(out[f'cls_{lname}'][b, 0]).cpu().numpy()
                obj_p = torch.sigmoid(out[f'obj_{lname}'][b, 0]).cpu().numpy()
                score = cls_p * obj_p
                bbox_map = out[f'bbox_{lname}'][b].cpu().numpy()
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(score, kernel)
                peaks = (score == dilated) & (score > 0.05)
                if not peaks.any():
                    continue
                for cy, cx in zip(*np.where(peaks)):
                    q = float(score[cy, cx])
                    dx = float(bbox_map[0, cy, cx])
                    dy = float(bbox_map[1, cy, cx])
                    dw = float(bbox_map[2, cy, cx])
                    dh = float(bbox_map[3, cy, cx])
                    box_cx = (cx + 0.5 + dx) * stride
                    box_cy = (cy + 0.5 + dy) * stride
                    bw = float(np.exp(np.clip(dw, -2, 5))) * stride
                    bh = float(np.exp(np.clip(dh, -2, 5))) * stride
                    if bw < 5 or bh < 5:
                        continue
                    all_dets.append((q, [box_cx - bw/2, box_cy - bh/2, bw, bh]))
            m = compute_map(all_dets, gt_boxes)
            total_map += m
            n += 1
    model.train()
    return total_map / max(n, 1)


# ── Training ──

def run():
    import argparse
    parser = argparse.ArgumentParser(description="FaceCNN V0 Training")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FaceCNNV0()
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        elif 'ema_state_dict' in state:
            model.load_state_dict(state['ema_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"Resumed: {args.resume}")
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"FaceCNN V0: {n:,} params")

    train_ds = WIDERFaceDataset(split="train")
    val_ds = WIDERFaceDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    cls_loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    obj_loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    bbox_loss_fn = EIoULoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    ema = ModelEMA(model, decay=0.999)

    csv_path = os.path.join(MODEL_DIR, "training_metrics_v0.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'loss', 'cls_loss', 'obj_loss', 'bbox_loss',
                     'lr', 'val_map', 'grad_norm', 'time_s'])

    best_map = 0.0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        losses, cls_l, obj_l, bbox_l, grad_norms = [], [], [], [], []

        for imgs, gts, n_gts in train_loader:
            imgs = imgs.to(device)
            gts = gts.to(device)
            n_gts = n_gts.to(device)
            feat_sizes = {
                '8': (imgs.shape[2] // 8, imgs.shape[3] // 8),
                '16': (imgs.shape[2] // 16, imgs.shape[3] // 16),
                '32': (imgs.shape[2] // 32, imgs.shape[3] // 32),
            }

            targets = build_targets(gts.cpu(), n_gts.cpu(), feat_sizes, [8, 16, 32])
            targets = {k: tuple(t.to(device) for t in targets[k]) for k in targets}

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = 0.0
                cl, ol, bl = 0.0, 0.0, 0.0
                for lname in ['8', '16', '32']:
                    cls_t, obj_t, bbox_t = targets[lname]
                    cls_p = torch.sigmoid(out[f'cls_{lname}'])
                    obj_p = torch.sigmoid(out[f'obj_{lname}'])
                    cl += cls_loss_fn(cls_p, cls_t)
                    ol += obj_loss_fn(obj_p, obj_t)
                    pos = (obj_t > 0).detach()
                    if pos.sum() > 0:
                        stride = {'8': 8, '16': 16, '32': 32}[lname]
                        bl += bbox_loss_fn(out[f'bbox_{lname}'], bbox_t, pos, stride)
                loss = cl + ol + bl * 5.0

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
            opt.zero_grad()

            losses.append(loss.item())
            cls_l.append(cl.item())
            obj_l.append(ol.item())
            bbox_l.append(bl.item() * 5.0)
            grad_norms.append(gn.item())
            ema.update(model)

        if epoch < args.warmup:
            for pg in opt.param_groups:
                pg['lr'] = args.lr * (epoch + 1) / args.warmup
        else:
            sched.step()

        val_map = validate(ema.ema, val_loader, device)

        avg_loss = sum(losses) / len(losses)
        avg_cl = sum(cls_l) / len(cls_l)
        avg_ol = sum(obj_l) / len(obj_l)
        avg_bl = sum(bbox_l) / len(bbox_l)
        avg_gn = sum(grad_norms) / len(grad_norms)
        current_lr = opt.param_groups[0]['lr']
        elapsed = time.time() - start

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={avg_loss:.3f} (cls={avg_cl:.3f} obj={avg_ol:.3f} bbox={avg_bl:.3f})  "
              f"val_map={val_map:.4f}  lr={current_lr:.2e}  gn={avg_gn:.2f}  "
              f"t={elapsed:.0f}s")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1, f"{avg_loss:.4f}", f"{avg_cl:.4f}", f"{avg_ol:.4f}",
                f"{avg_bl:.4f}", f"{current_lr:.2e}", f"{val_map:.4f}",
                f"{avg_gn:.4f}", f"{elapsed:.0f}"])

        if val_map > best_map:
            best_map = val_map
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.ema.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'val_map': val_map,
            }, os.path.join(MODEL_DIR, "face_cnn_v0_best.pth"))
            print(f"  -> saved best (mAP={val_map:.4f})")

    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.ema.state_dict(),
        'val_map': val_map,
    }, os.path.join(MODEL_DIR, "face_cnn_v0_final.pth"))
    print(f"\nDone. Best mAP={best_map:.4f}")

    model.eval()
    ema.ema.eval()
    dummy = torch.randn(1, 3, 640, 640).to(device)
    onnx_path = os.path.join(MODEL_DIR, "face_cnn_v0.onnx")
    torch.onnx.export(ema.ema, dummy, onnx_path,
        input_names=['input'],
        output_names=['cls_8', 'obj_8', 'bbox_8',
                      'cls_16', 'obj_16', 'bbox_16',
                      'cls_32', 'obj_32', 'bbox_32'],
        opset_version=11)
    print(f"ONNX: {onnx_path}")


if __name__ == "__main__":
    run()
