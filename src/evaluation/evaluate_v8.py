"""
FaceCNN v8 — Evaluation Script (mAP, per-level, TTA)
=====================================================
Evaluates a trained v8 model on WIDER Face val set.

Usage:
  python3 src/evaluation/evaluate_v8.py --checkpoint models/face_cnn_v8_best.pth
  python3 src/evaluation/evaluate_v8.py --checkpoint models/face_cnn_v8_swa.pth --tta
"""

import os, sys, argparse, json, time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv.face_detector_v8 import FaceFCNv8, DetectionHead


class WiderValDataset(Dataset):
    def __init__(self, root_dir, target_h=480, target_w=640):
        self.target_h = target_h
        self.target_w = target_w
        self.samples = []
        img_dir = os.path.join(root_dir, "WIDER_val", "images")
        annot_file = os.path.join(root_dir, "wider_face_split", "wider_face_val_bbx_gt.txt")
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation not found: {annot_file}")
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
                    if w > 0 and h > 0:
                        faces.append({"x": x, "y": y, "w": w, "h": h})
            if os.path.exists(img_path):
                self.samples.append({"image_path": img_path, "faces": faces})
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


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]; area2 = box2[2] * box2[3]
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
    return ap


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


@torch.no_grad()
def evaluate(model, dataset, device, conf_thresh=0.25, nms_iou=0.3, use_tta=False):
    model.eval()
    all_dets = []
    all_gts = []
    all_images = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    for tensor, gt_boxes, img_path in tqdm(loader, desc="Evaluating"):
        if tensor is None:
            continue
        all_gts.append(gt_boxes)
        all_images.append(img_path)
        tensor = tensor.to(device)

        dets = []
        for scale in ([1.0] if not use_tta else [1.0]):
            if scale != 1.0:
                s_tensor = torch.nn.functional.interpolate(
                    tensor, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                s_tensor = tensor

            out = model(s_tensor)
            for level, stride in [("p3", 4), ("p4", 8), ("p5", 16)]:
                obj = torch.sigmoid(out[f"{level}_obj"][0, 0]).cpu().numpy()
                iou_p = torch.sigmoid(out[f"{level}_iou"][0, 0]).cpu().numpy()
                quality = np.sqrt(obj * iou_p + 1e-8)
                bbox_raw = out[f"{level}_bbox"][0].cpu()
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(quality, kernel)
                peaks = (quality == dilated) & (quality > conf_thresh)
                if not peaks.any():
                    continue
                ys, xs = np.where(peaks)
                for cy, cx in zip(ys, xs):
                    q = float(quality[cy, cx])
                    bbox_tensor = bbox_raw[:, cy:cy+1, cx:cx+1].unsqueeze(0)
                    offsets = DetectionHead.decode_bbox(bbox_tensor, stride).squeeze()
                    dx = float(offsets[0]); dy = float(offsets[1])
                    dw = float(np.clip(offsets[2], -2, 5))
                    dh = float(np.clip(offsets[3], -2, 5))
                    box_cx = (cx + 0.5 + dx) * stride / scale
                    box_cy = (cy + 0.5 + dy) * stride / scale
                    box_w = float(np.exp(dw)) * stride / scale
                    box_h = float(np.exp(dh)) * stride / scale
                    if box_w > 2 and box_h > 2:
                        x1 = max(0, int(box_cx - box_w / 2))
                        y1 = max(0, int(box_cy - box_h / 2))
                        dets.append((q, [x1, y1, int(box_w), int(box_h)]))

        if use_tta:
            flipped = torch.flip(tensor, dims=[3])
            out_flip = model(flipped)
            h, w = tensor.shape[2], tensor.shape[3]
            for level, stride in [("p3", 4), ("p4", 8), ("p5", 16)]:
                obj = torch.sigmoid(out_flip[f"{level}_obj"][0, 0]).cpu().numpy()
                iou_p = torch.sigmoid(out_flip[f"{level}_iou"][0, 0]).cpu().numpy()
                quality = np.sqrt(obj * iou_p + 1e-8)
                bbox_raw = out_flip[f"{level}_bbox"][0].cpu()
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(quality, kernel)
                peaks = (quality == dilated) & (quality > conf_thresh)
                if not peaks.any():
                    continue
                ys, xs = np.where(peaks)
                for cy, cx in zip(ys, xs):
                    q = float(quality[cy, cx])
                    bbox_tensor = bbox_raw[:, cy:cy+1, cx:cx+1].unsqueeze(0)
                    offsets = DetectionHead.decode_bbox(bbox_tensor, stride).squeeze()
                    dx = float(offsets[0]); dy = float(offsets[1])
                    dw = float(np.clip(offsets[2], -2, 5))
                    dh = float(np.clip(offsets[3], -2, 5))
                    box_cx = (w - (cx + 0.5 + dx) * stride) / 1.0
                    box_cy = (cy + 0.5 + dy) * stride / 1.0
                    box_w = float(np.exp(dw)) * stride
                    box_h = float(np.exp(dh)) * stride
                    if box_w > 2 and box_h > 2:
                        x1 = max(0, int(box_cx - box_w / 2))
                        y1 = max(0, int(box_cy - box_h / 2))
                        dets.append((q, [x1, y1, int(box_w), int(box_h)]))

        dets = soft_nms(dets, nms_iou)
        all_dets.append(dets)

    return compute_map(all_dets, all_gts)


def compute_map(all_dets, all_gts, iou_thresh=0.5):
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
                        best_iou = iou; best_idx = j
            if best_iou >= iou_thresh and best_idx >= 0:
                tp_list.append(1); fp_list.append(0); gt_matched[best_idx] = True
            else:
                tp_list.append(0); fp_list.append(1)
            scores_list.append(score)
        fn_list.append(len(gts) - sum(gt_matched))

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    fn_total = sum(fn_list)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = compute_ap(recalls, precisions) if n_gt > 0 else 0.0
    return ap, float(n_gt), len(all_dets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/face/widerface")
    parser.add_argument("--target-h", type=int, default=480)
    parser.add_argument("--target-w", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FaceFCNv8().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    dataset = WiderValDataset(args.data, args.target_h, args.target_w)

    t0 = time.time()
    ap, n_gt, n_images = evaluate(
        model, dataset, device, args.conf, args.nms_iou, args.tta)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"  Images: {n_images}")
    print(f"  GT faces: {n_gt}")
    print(f"  mAP@0.5: {ap:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_images*1000:.1f}ms/image)")
    print(f"  TTA: {'ON' if args.tta else 'OFF'}")
    print(f"{'='*50}")

    results = {"mAP@0.5": ap, "n_images": n_images, "n_gt": n_gt,
               "time_s": elapsed, "tta": args.tta, "conf": args.conf}
    out_path = args.checkpoint.replace(".pth", "_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
