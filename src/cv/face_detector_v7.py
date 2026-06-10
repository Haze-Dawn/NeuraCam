"""
FaceFCN v7.0 — Ground-Up Redesign
==================================

Parameter budget: 499K
Allocation: Backbone 41% | FPN 12% | Heads 47%

Design: Max 128ch backbone (192ch pointwise convs are 36K each — too expensive).
Deep backbone with 16 DSConv blocks total. 4-layer detection heads (256->192->128
hidden dims). DSConv FPN refine. SE channel attention. Label smoothing + bbox loss
weighting. No P2 level.

Unlike v6 (0.2% params on heads), v7 gives heads 47% of params — 230x more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                    padding=dilation, dilation=dilation,
                                    groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(4, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x)


class V7Backbone(nn.Module):
    """16 DSConv blocks, max 128ch. 3 output levels (C3/C4/C5)."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.s1 = nn.Sequential(
            DSConvBlock(32, 48),
            DSConvBlock(48, 64),
        )
        self.s2_down = DSConvBlock(64, 96, stride=2)
        self.s2 = nn.Sequential(*[DSConvBlock(96, 96) for _ in range(5)])
        self.s3_down = DSConvBlock(96, 128, stride=2)
        self.s3 = nn.Sequential(*[DSConvBlock(128, 128) for _ in range(5)])
        self.s4 = nn.Sequential(
            DSConvBlock(128, 128, dilation=2),
            DSConvBlock(128, 128, dilation=2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        s1 = self.s1(x)
        s2 = self.s2_down(s1)
        c3 = self.s2(s2)
        s3 = self.s3_down(c3)
        c4 = self.s3(s3)
        c5 = self.s4(c4)
        return c3, c4, c5


class V7FPN(nn.Module):
    """96ch FPN with depthwise-separable refine convs (saves 64K vs regular)."""
    def __init__(self, fpn_dim=96):
        super().__init__()
        self.lat3 = nn.Conv2d(96, fpn_dim, 1)
        self.lat4 = nn.Conv2d(128, fpn_dim, 1)
        self.lat5 = nn.Conv2d(128, fpn_dim, 1)

        self.refine3 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True),
        )
        self.refine4 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True),
        )
        self.se3 = SEBlock(fpn_dim)
        self.se4 = SEBlock(fpn_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:],
                                            mode='bilinear', align_corners=False)
        p4 = self.refine4(p4)
        p4 = self.se4(p4) + p4

        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:],
                                            mode='bilinear', align_corners=False)
        p3 = self.refine3(p3)
        p3 = self.se3(p3) + p3
        return p3, p4


class V7SharedHead(nn.Module):
    """Shared detection head: 2 shared conv layers + per-level branches.

    Shared layers (trained on ALL scales):
      conv1: 3x3 Conv(96→192) — spatial context, 9×9 receptive field
      conv2: 1x1 Conv(192→192) — channel mixing

    Per-level branches (independent P3/P4, specialized per stride):
      conv3: 1x1 Conv(192→96) — project
      conv4: 1x1 Conv(96→64)  — compress
      obj_pred: 64→1 — objectness logit
      iou_pred: 64→1 — estimated IoU with GT (quality score for NMS)
      bbox_pred: 64→4 — bbox offsets

    The shared layers see ~13 positives per image (P3=10 + P4=3), vs
    v6 where P4's independent head saw only ~3. This 4.3× gradient
    amplification prevents the head collapse seen in v6.

    The IoU branch predicts the IoU between predicted box and GT.
    At inference, the IoU score is used to rank detections instead
    of obj confidence — FCOS proved this improves NMS quality.
    """
    def __init__(self, in_dim=96, shared_dim=192, hid_dim=96, pred_dim=64,
                 p3_enhanced=False):
        super().__init__()
        self.p3_enhanced = p3_enhanced
        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(in_dim, shared_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )
        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )

        self.p3_conv3 = nn.Sequential(
            nn.Conv2d(shared_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim), nn.ReLU(inplace=True),
        )
        self.p3_conv4 = nn.Sequential(
            nn.Conv2d(hid_dim, pred_dim, 1, bias=False),
            nn.BatchNorm2d(pred_dim), nn.ReLU(inplace=True),
        )

        if p3_enhanced:
            self.p3_project_up = nn.Sequential(
                nn.Conv2d(pred_dim, 128, 1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.p3_se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 128, 1),
                nn.Sigmoid(),
            )
            self.p3_refine = nn.Sequential(
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, groups=8, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.p3_obj = nn.Conv2d(128, 1, 3, padding=1)
            self.p3_iou = nn.Conv2d(128, 1, 3, padding=1)
            self.p3_bbox = nn.Conv2d(128, 4, 3, padding=1)
            self.p3_cell_bias = nn.Parameter(torch.zeros(1, 1, 120, 160))

        else:
            self.p3_obj = nn.Conv2d(pred_dim, 1, 1)
            self.p3_iou = nn.Conv2d(pred_dim, 1, 1)
            self.p3_bbox = nn.Conv2d(pred_dim, 4, 1)

        self.p4_conv3 = nn.Sequential(
            nn.Conv2d(shared_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim), nn.ReLU(inplace=True),
        )
        self.p4_conv4 = nn.Sequential(
            nn.Conv2d(hid_dim, pred_dim, 1, bias=False),
            nn.BatchNorm2d(pred_dim), nn.ReLU(inplace=True),
        )
        self.p4_obj = nn.Conv2d(pred_dim, 1, 1)
        self.p4_iou = nn.Conv2d(pred_dim, 1, 1)
        self.p4_bbox = nn.Conv2d(pred_dim, 4, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.p3_obj.bias, -2.5)
        nn.init.constant_(self.p4_obj.bias, -2.5)
        nn.init.zeros_(self.p3_iou.bias)
        nn.init.zeros_(self.p4_iou.bias)
        nn.init.zeros_(self.p3_bbox.bias)
        nn.init.zeros_(self.p4_bbox.bias)
        self.p3_bbox.bias.data[2:] = -2.0
        self.p4_bbox.bias.data[2:] = -2.0

    def forward(self, p3_feat, p4_feat):
        p3 = self.shared_conv2(self.shared_conv1(p3_feat))
        p4 = self.shared_conv2(self.shared_conv1(p4_feat))

        p3 = self.p3_conv4(self.p3_conv3(p3))
        p4 = self.p4_conv4(self.p4_conv3(p4))

        if self.p3_enhanced:
            p3 = self.p3_project_up(p3)  # 64→128
            p3 = p3 * self.p3_se(p3)     # SE gating
            p3 = self.p3_refine(p3)      # pointwise + grouped conv

        return {
            "p3_obj": self.p3_obj(p3) + (self.p3_cell_bias if self.p3_enhanced else 0),
            "p3_iou": self.p3_iou(p3),
            "p3_bbox": self.p3_bbox(p3),
            "p4_obj": self.p4_obj(p4),
            "p4_iou": self.p4_iou(p4),
            "p4_bbox": self.p4_bbox(p4),
        }


class FaceFCNv7(nn.Module):
    """Shared-head face detector. P3 (stride 4) + P4 (stride 8).
    Output includes IoU quality branch for NMS filtering."""
    strides = {"p3": 4, "p4": 8}

    def __init__(self, p3_enhanced=False):
        super().__init__()
        self.backbone = V7Backbone()
        self.fpn = V7FPN(fpn_dim=96)
        self.head = V7SharedHead(in_dim=96, shared_dim=192, hid_dim=96, pred_dim=64,
                                  p3_enhanced=p3_enhanced)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4 = self.fpn(c3, c4, c5)
        out = self.head(p3, p4)
        # Add backward-compatible obj/bbox keys for training
        out.update({
            "p3_obj": out["p3_obj"],
            "p3_bbox": out["p3_bbox"],
            "p4_obj": out["p4_obj"],
            "p4_bbox": out["p4_bbox"],
        })
        return out

    @staticmethod
    def compute_quality(raw_obj, raw_iou):
        """Compute detection quality = sqrt(sigmoid(obj) * sigmoid(iou)).
        
        Quality score combines objectness confidence with IoU quality.
        A detection that's confident AND well-localized scores higher
        than one that's confident but poorly placed. Used for thresholding
        and NMS ranking at inference.
        
        Returns quality in [0, 1].
        """
        obj_p = torch.sigmoid(raw_obj)
        iou_p = torch.sigmoid(raw_iou)
        return torch.sqrt(obj_p * iou_p + 1e-8)
    
    @torch.no_grad()
    def detect(self, frame, conf_thresholds=None, nms_iou=0.3):
        """Full detection pipeline: forward → peak-finding → decode → NMS.
        
        Uses quality = sqrt(sigmoid(obj) * sigmoid(iou)) instead of
        raw obj confidence for thresholding and NMS ranking.
        
        Args:
            frame: BGR numpy array (H, W, 3)
            conf_thresholds: dict of {level: quality_threshold}, e.g.
                {"p3": 0.30, "p4": 0.10}. If None, uses 0.25 for all.
            nms_iou: IoU threshold for greedy NMS (default 0.3)
        Returns:
            list of (level, quality, BoundingBox) tuples
        """
        import cv2
        from src.cv.face_tracker import BoundingBox, compute_iou
        
        if conf_thresholds is None:
            conf_thresholds = {"p3": 0.25, "p4": 0.25}
        
        device = next(self.parameters()).device
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        
        out = self.forward(tensor.unsqueeze(0).to(device))
        
        levels = [
            ("p3", 4, out["p3_obj"][0, 0], out["p3_iou"][0, 0], out["p3_bbox"][0]),
            ("p4", 8, out["p4_obj"][0, 0], out["p4_iou"][0, 0], out["p4_bbox"][0]),
        ]
        
        all_dets = []
        for lname, stride, obj_map, iou_map, bbox_map in levels:
            quality = self.compute_quality(obj_map, iou_map).cpu().numpy()
            thresh = conf_thresholds.get(lname, 0.25)
            
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(quality, kernel)
            peaks = (quality == dilated) & (quality > thresh)
            if not peaks.any():
                continue
            
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
                all_dets.append((
                    lname, q, BoundingBox(x=x1, y=y1, w=bw, h=bh)
                ))
        
        # Soft-NMS: sort by quality descending, decay overlapping
        # Instead of hard suppression (binary keep/discard), decay the
        # quality score by (1 - IoU). A detection with quality 0.90
        # that overlaps another by IoU=0.8 becomes 0.90 * (1-0.8) = 0.18.
        # This preserves valid detections for nearby faces while still
        # ranking unique faces higher.
        all_dets.sort(key=lambda x: x[1], reverse=True)
        kept = []
        for det in all_dets:
            max_decay = 0.0
            for k in kept:
                iou = compute_iou(det[2], k[2])
                if iou > nms_iou:
                    max_decay = max(max_decay, iou)
            if max_decay > 0:
                # Soft decay: quality *= (1 - IoU)
                decayed = det[1] * (1.0 - max_decay)
                if decayed > 0.01:
                    kept.append((det[0], decayed, det[2]))
            else:
                kept.append(det)
        return kept


if __name__ == "__main__":
    model = FaceFCNv7()
    total = sum(p.numel() for p in model.parameters())

    bb = sum(p.numel() for n, p in model.named_parameters() if n.startswith('backbone'))
    fp = sum(p.numel() for n, p in model.named_parameters() if n.startswith('fpn'))
    hd = sum(p.numel() for n, p in model.named_parameters() if n.startswith('head'))

    sf = sum(p.numel() for n, p in model.named_parameters() if n.startswith('head.shared'))
    br = sum(p.numel() for n, p in model.named_parameters() if n.startswith('head.p'))

    print(f"FaceFCNv7 — Shared-Head + IoU + Deep Branches")
    print(f"{'='*55}")
    print(f"  Backbone:  {bb:>8,} ({100*bb/total:.0f}%)")
    print(f"  FPN:       {fp:>8,} ({100*fp/total:.0f}%)")
    print(f"  Heads:     {hd:>8,} ({100*hd/total:.0f}%)")
    print(f"    Shared:  {sf:>8,}    (3×3 conv1 + 1×1 conv2, 192ch)")
    print(f"    Branches:{br:>8,}    (conv3+conv4+obj+iou+bbox, ×2 levels)")
    print(f"  {'─'*35}")
    print(f"  Total:     {total:>8,}")
    print(f"  Budget:    500,000")
    print(f"  Remaining: {500000 - total:,}")
    print(f"{'='*55}")

    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        out = model(x)
    for k, v in out.items():
        print(f"  {k}: {list(v.shape)}")
