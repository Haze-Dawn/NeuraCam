"""
FaceFCN v7.0 — P4-Only Inference Model
========================================
Stripped of all P3 (stride 4) components. Uses only P4 (stride 8) for
detection. Backbone and FPN are unchanged — only the detection head
is simplified to remove P3 branches.

Total params: ~453K (vs 519K with P3)
Inference speed: ~90-110% of the full model (FPN still computes P3 features,
but head is smaller and P4 output has 4× fewer cells to decode).

Checkpoint: face_cnn_v7.pth (Phase 1 best, ep 143, P4 F1=0.611)
Use export_p4_checkpoint() to create a stripped checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.cv.face_detector_v7 import V7Backbone, V7FPN


class P4OnlyHead(nn.Module):
    """Detection head with P4 only — no P3 branches at all.
    
    Architecture (29,880 params):
      conv1: 3×3 Conv(96→192) — shared spatial context
      conv2: 1×1 Conv(192→192) — shared channel mixing
      conv3: 1×1 Conv(192→96) — P4 project
      conv4: 1×1 Conv(96→64)  — P4 compress
      obj: 64→1 — objectness logit
      iou: 64→1 — quality score
      bbox: 64→4 — center offsets + log-w/h
    """
    def __init__(self, in_dim=96, shared_dim=192, hid_dim=96, pred_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, shared_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(shared_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim), nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hid_dim, pred_dim, 1, bias=False),
            nn.BatchNorm2d(pred_dim), nn.ReLU(inplace=True),
        )
        self.obj = nn.Conv2d(pred_dim, 1, 1)
        self.iou = nn.Conv2d(pred_dim, 1, 1)
        self.bbox = nn.Conv2d(pred_dim, 4, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.obj.bias, -2.5)
        nn.init.zeros_(self.iou.bias)
        nn.init.zeros_(self.bbox.bias)
        self.bbox.bias.data[2:] = -2.0

    def forward(self, p4_feat):
        x = self.conv2(self.conv1(p4_feat))
        x = self.conv4(self.conv3(x))
        return {
            "p4_obj": self.obj(x),
            "p4_iou": self.iou(x),
            "p4_bbox": self.bbox(x),
        }


class FaceFCNv7P4Only(nn.Module):
    """P4-only face detector — no P3 heads.
    
    Outputs: p4_obj (objectness logits), p4_iou (quality scores),
             p4_bbox (bbox offsets) at stride 8.
    """
    strides = {"p4": 8}

    def __init__(self):
        super().__init__()
        self.backbone = V7Backbone()
        self.fpn = V7FPN(fpn_dim=96)
        self.head = P4OnlyHead(in_dim=96, shared_dim=192, hid_dim=96, pred_dim=64)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        _, p4 = self.fpn(c3, c4, c5)
        return self.head(p4)

    @staticmethod
    def compute_quality(raw_obj, raw_iou):
        quality = torch.sqrt(torch.sigmoid(raw_obj) * torch.sigmoid(raw_iou) + 1e-8)
        return quality

    @torch.no_grad()
    def detect(self, frame, conf_threshold=0.15, nms_iou=0.3):
        """Full P4 detection pipeline: forward → peak-finding → decode → NMS.
        
        P4-only: 60×80 grid at stride 8. Use lower conf_threshold (0.15)
        vs the full model's 0.25 because P3 is removed and all small-face
        detection responsibility falls on P4.
        
        Args:
            frame: BGR numpy array (H, W, 3)
            conf_threshold: quality threshold for P4 (default 0.15)
            nms_iou: IoU threshold for greedy NMS (default 0.3)
        Returns:
            list of (level, quality, BoundingBox) tuples
        """
        import cv2
        from src.cv.face_tracker import BoundingBox, compute_iou

        device = next(self.parameters()).device
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        out = self.forward(tensor.unsqueeze(0).to(device))

        stride = 8
        obj_map = out["p4_obj"][0, 0]
        iou_map = out["p4_iou"][0, 0]
        bbox_map = out["p4_bbox"][0]

        quality = self.compute_quality(obj_map, iou_map).cpu().numpy()

        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(quality, kernel)
        peaks = (quality == dilated) & (quality > conf_threshold)
        
        all_dets = []
        if peaks.any():
            ys, xs = np.where(peaks)
            for cy, cx in zip(ys, xs):
                q = float(quality[cy, cx])
                dx = float(bbox_map[0, cy, cx].item())
                dy = float(bbox_map[1, cy, cx].item())
                dw = float(bbox_map[2, cy, cx].item())
                dh = float(bbox_map[3, cy, cx].item())
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
                all_dets.append(("p4", q, BoundingBox(x=x1, y=y1, w=bw, h=bh)))

        all_dets.sort(key=lambda x: x[1], reverse=True)
        kept = []
        for det in all_dets:
            max_decay = 0.0
            for k in kept:
                iou = compute_iou(det[2], k[2])
                if iou > nms_iou:
                    max_decay = max(max_decay, iou)
            if max_decay > 0:
                decayed = det[1] * (1.0 - max_decay)
                if decayed > 0.01:
                    kept.append((det[0], decayed, det[2]))
            else:
                kept.append(det)
        return kept


def export_p4_checkpoint(source_path, output_path):
    """Load Phase 1 checkpoint and save a P4-only checkpoint.
    
    Removes all head.p3_* keys and creates a P4OnlyHead-compatible state dict
    by prefixing head.p4_* → head.obj/iou/bbox.
    """
    ckpt = torch.load(source_path, map_location='cpu')
    state = ckpt['model_state_dict']
    
    p4_state = {}
    for k, v in state.items():
        if k.startswith('backbone') or k.startswith('fpn'):
            p4_state[k] = v
        elif k.startswith('head.shared'):
            # Rename head.shared_conv* → head.conv*
            new_k = k.replace('head.shared_conv', 'head.conv')
            p4_state[new_k] = v
        elif k.startswith('head.p4_'):
            # Rename head.p4_* → head.* (strip p4_ prefix)
            new_k = k.replace('head.p4_', 'head.')
            p4_state[new_k] = v
    
    # Create model and load
    model = FaceFCNv7P4Only()
    missing, unexpected = model.load_state_dict(p4_state, strict=False)
    
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    if missing:
        print(f"Missing keys: {missing}")
    
    # Save checkpoint
    torch.save({
        'epoch': ckpt.get('epoch', 0),
        'model_state_dict': model.state_dict(),
        'p4_f1': ckpt.get('best_val_f1', 0.0),
        'architecture': 'FaceFCNv7P4Only',
        'description': 'P4-only detector stripped of P3 heads. '
                       'Exported from Phase 1 checkpoint epoch 143.',
    }, output_path)
    print(f"P4-only checkpoint saved: {output_path}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify inference
    x = torch.randn(1, 3, 480, 640)
    out = model(x)
    print(f"Output: p4_obj={list(out['p4_obj'].shape)}, "
          f"p4_iou={list(out['p4_iou'].shape)}, "
          f"p4_bbox={list(out['p4_bbox'].shape)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        export_p4_checkpoint(sys.argv[1], sys.argv[2])
    else:
        source = "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo/models/face_cnn_v7.pth"
        output = "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo/models/face_cnn_v7_p4only.pth"
        export_p4_checkpoint(source, output)
