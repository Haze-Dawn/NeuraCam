"""
FaceFCN v8.0 — Independent-Head, BiFPN, Varifocal Architecture
================================================================
Target: 0.72-0.78 mAP on WIDER Face val, <35ms CPU inference

Key architectural decisions (fixing V7's unsolvable problems):
  1. Independent per-level heads — each level trains on its own stride's
     positives. No shared weights = no gradient cancellation (the P3 killer).
  2. BiFPN neck — weighted feature fusion with top-down + bottom-up paths.
  3. Deeper backbone (21 blocks), max 192ch with HardSwish activations.
  4. VarifocalLoss + EIoU + DFL for training.
  5. Distribution Focal Loss for bbox — predicts a distribution over 16 bins
     per offset rather than a single scalar (proven +1-2 mAP on COCO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


# ──────────────────────────────────────────────
# Utility Modules
# ──────────────────────────────────────────────

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, inplace=self.inplace) / 6.0


class DSConvBlock(nn.Module):
    """Depthwise Separable Conv block with HardSwish activation."""
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                    padding=dilation, dilation=dilation,
                                    groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = HardSwish(inplace=True)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = HardSwish(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
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


class WeightedFusion(nn.Module):
    """Learnable weighted fusion for BiFPN.
    
    Learns 2 per-channel weights per level: one for the lateral/skip
    feature, one for the top-down/bottom-up feature. Softmax-normalized
    so weights are always positive and sum to 1.
    """
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, channels))

    def forward(self, feat_a, feat_b):
        w = F.softmax(self.weights, dim=0)
        return w[0:1, :, None, None] * feat_a + w[1:2, :, None, None] * feat_b


# ──────────────────────────────────────────────
# Backbone: V8Backbone
# ──────────────────────────────────────────────

class V8Backbone(nn.Module):
    """21 depthwise separable conv blocks, max 192ch, HardSwish.
    
    Outputs 3 feature levels for the FPN:
      C3: stride 4,  128ch, 120×160 — small face features
      C4: stride 8,  160ch,  60×80  — medium face features
      C5: stride 16, 192ch,  30×40  — large face features + top-down source
    
    Depth allocation (blocks per stage): 3, 6, 6, 6 = 21 blocks
    Channel progression: 32→64→128→160→192
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            HardSwish(inplace=True),
        )

        # Stage 1: stride 2 → 4, 3 blocks, 32→64→64
        self.s1 = nn.Sequential(
            DSConvBlock(32, 64, stride=2),
            DSConvBlock(64, 64),
            DSConvBlock(64, 64),
        )

        # Stage 2: stride 1 (stay at 4), 6 blocks, 64→128→128×5
        self.s2 = nn.Sequential(
            DSConvBlock(64, 128),
            DSConvBlock(128, 128),
            DSConvBlock(128, 128),
            DSConvBlock(128, 128),
            DSConvBlock(128, 128),
            DSConvBlock(128, 128),
        )

        # Stage 3: stride 2 → 8, 6 blocks, 128→160→160×5
        self.s3 = nn.Sequential(
            DSConvBlock(128, 160, stride=2),
            DSConvBlock(160, 160),
            DSConvBlock(160, 160),
            DSConvBlock(160, 160),
            DSConvBlock(160, 160),
            DSConvBlock(160, 160),
        )

        # Stage 4: stride 2 → 16, 6 blocks, 160→192→192×4, dilation=2
        self.s4 = nn.Sequential(
            DSConvBlock(160, 192, stride=2),
            DSConvBlock(192, 192, dilation=2),
            DSConvBlock(192, 192, dilation=2),
            DSConvBlock(192, 192, dilation=2),
            DSConvBlock(192, 192, dilation=2),
            DSConvBlock(192, 192, dilation=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        c2 = self.stem(x)              # 240×320, stride 2
        x = self.s1(c2)                # 120×160, stride 4
        x = self.s2(x)                 # 120×160, stride 4 ← C3
        c3 = x
        x = self.s3(x)                 #  60×80,  stride 8 ← C4
        c4 = x
        x = self.s4(x)                 #  30×40,  stride 16 ← C5
        c5 = x
        return c2, c3, c4, c5


# ──────────────────────────────────────────────
# BiFPN Neck
# ──────────────────────────────────────────────

class BiFPN(nn.Module):
    """Weighted BiFPN with top-down + bottom-up paths, 5 levels (P2-P6).
    
    P2 is used for context only (not a detection level).
    Detection levels: P3 (stride 4), P4 (stride 8), P5 (stride 16).
    
    Architecture:
      Lateral convs: 1×1 projects backbone features to fpn_dim=96.
      Top-down: P5 → P4 → P3 → P2 with weighted fusion + refine conv.
      Bottom-up: P2 → P3 → P4 → P5 → P6 with weighted fusion + refine conv.
      P6 is a stride-2 conv on P5 for global context.
    """
    def __init__(self, fpn_dim=104):
        super().__init__()
        # Lateral connections: project backbone channels to fpn_dim
        self.lat2 = nn.Conv2d(32, fpn_dim, 1)   # from C2 (stem out, 32ch stride 2)
        self.lat3 = nn.Conv2d(128, fpn_dim, 1)  # from C3 (stage 2 out)
        self.lat4 = nn.Conv2d(160, fpn_dim, 1)  # from C4 (stage 3 out)
        self.lat5 = nn.Conv2d(192, fpn_dim, 1)  # from C5 (stage 4 out)

        # Top-down weighted fusion
        self.fuse_td_p4 = WeightedFusion(fpn_dim)  # lat4 + upsample(P5)
        self.fuse_td_p3 = WeightedFusion(fpn_dim)  # lat3 + upsample(P4_td)
        self.fuse_td_p2 = WeightedFusion(fpn_dim)  # lat2 + upsample(P3_td)

        # Bottom-up weighted fusion
        self.fuse_bu_p3 = WeightedFusion(fpn_dim)  # P3_td + downsample(P2_bu)
        self.fuse_bu_p4 = WeightedFusion(fpn_dim)  # P4_td + downsample(P3_bu)
        self.fuse_bu_p5 = WeightedFusion(fpn_dim)  # P5_td + downsample(P4_bu)

        # SE attention gates per level (with residual connection)
        self.se3 = SEBlock(fpn_dim)
        self.se4 = SEBlock(fpn_dim)
        self.se5 = SEBlock(fpn_dim)

        # Depthwise-separable refine convs (anti-aliasing after up/down sample)
        def refine_block():
            return nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1,
                          groups=fpn_dim, bias=False),
                nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                HardSwish(inplace=True),
            )

        self.refine_td_p4 = refine_block()
        self.refine_td_p3 = refine_block()
        self.refine_td_p2 = refine_block()
        self.refine_bu_p3 = refine_block()
        self.refine_bu_p4 = refine_block()
        self.refine_bu_p5 = refine_block()

        # P6: global context level (stride 32)
        self.p6_conv = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, stride=2, padding=1,
                      groups=fpn_dim, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            HardSwish(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c2, c3, c4, c5):
        """c2 (64ch, 240×320), c3 (128ch, 120×160), c4 (160ch, 60×80), c5 (192ch, 30×40).
        Returns p3_out, p4_out, p5_out at fpn_dim for detection heads.
        """
        # Lateral projections: backbone → FPN dim
        p2_lat = self.lat2(c2)   # 96×240×320
        p3_lat = self.lat3(c3)   # 96×120×160
        p4_lat = self.lat4(c4)   # 96×60×80
        p5_lat = self.lat5(c5)   # 96×30×40

        # ── Top-down pass ──
        p5_td = p5_lat                                           # 96×30×40
        p4_td = self.refine_td_p4(
            self.fuse_td_p4(p4_lat,
                            F.interpolate(p5_td, size=c4.shape[-2:],
                                          mode='bilinear', align_corners=False))
        )                                                        # 96×60×80
        p3_td = self.refine_td_p3(
            self.fuse_td_p3(p3_lat,
                            F.interpolate(p4_td, size=c3.shape[-2:],
                                          mode='bilinear', align_corners=False))
        )                                                        # 96×120×160
        p2_td = self.refine_td_p2(
            self.fuse_td_p2(p2_lat,
                            F.interpolate(p3_td, size=c2.shape[-2:],
                                          mode='bilinear', align_corners=False))
        )                                                        # 96×240×320

        # ── Bottom-up pass ──
        # P2_bu = P2_td (no upsample needed — bottom-up starts here)
        p2_bu = p2_td                                            # 96×60×80
        p3_bu = self.refine_bu_p3(
            self.fuse_bu_p3(p3_td,
                            F.interpolate(p2_bu, size=c3.shape[-2:],
                                          mode='bilinear', align_corners=False))
        )                                                        # 96×120×160
        p4_bu = self.refine_bu_p4(
            self.fuse_bu_p4(p4_td,
                            F.interpolate(p3_bu, size=c4.shape[-2:],
                                          mode='bilinear', align_corners=False))
        )                                                        # 96×60×80
        p5_bu = self.refine_bu_p5(
            self.fuse_bu_p5(p5_td,
                            F.interpolate(p4_bu, size=c5.shape[-2:],
                                          mode='bilinear', align_corners=False))
        )                                                        # 96×30×40

        # P6: global context
        p6 = self.p6_conv(p5_bu)                                 # 96×15×20

        # SE attention gates with residual
        p3_out = self.se3(p3_bu) + p3_bu                         # 96×120×160
        p4_out = self.se4(p4_bu) + p4_bu                         # 96×60×80
        p5_out = self.se5(p5_bu) + p5_bu                         # 96×30×40

        return p3_out, p4_out, p5_out


# ──────────────────────────────────────────────
# Independent Detection Heads
# ──────────────────────────────────────────────

class DetectionHead(nn.Module):
    """Per-level detection head. INDEPENDENT — no weight sharing between levels.
    
    3-layer MLP (all 1×1 convs for efficiency):
      conv1: 96→96, BN, HardSwish
      conv2: 96→96, BN, HardSwish
      obj:   96→1     — quality score logit
      iou:   96→1     — IoU prediction for NMS ranking
      bbox:  96→4×16  — Distribution Focal Loss bbox (4 offsets × 16 bins)
    
    Bias init per level varies (denser grids need lower prior):
      P3 (stride 4,  19,200 cells): bias = -3.0  → sigmoid ≈ 0.047
      P4 (stride 8,   4,800 cells): bias = -2.5  → sigmoid ≈ 0.076
      P5 (stride 16,  1,200 cells): bias = -2.5  → sigmoid ≈ 0.076
    """
    DFL_BINS = 16

    def __init__(self, name, in_dim=104, hid_dim=104, obj_bias=-2.5):
        super().__init__()
        self.name = name
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=True),
        )
        self.obj = nn.Conv2d(hid_dim, 1, 1)
        self.iou = nn.Conv2d(hid_dim, 1, 1)
        self.bbox = nn.Conv2d(hid_dim, 4 * self.DFL_BINS, 1)
        self._init_weights(obj_bias)

    def _init_weights(self, obj_bias):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.obj.bias, obj_bias)
        nn.init.zeros_(self.iou.bias)
        nn.init.zeros_(self.bbox.bias)
        self.bbox.bias.data[2 * self.DFL_BINS:] = -2.0  # dw, dh start small

    def forward(self, feat):
        x = self.conv3(self.conv2(self.conv1(feat)))
        return {
            f"{self.name}_obj": self.obj(x),
            f"{self.name}_iou": self.iou(x),
            f"{self.name}_bbox": self.bbox(x),
        }

    @staticmethod
    def decode_bbox(bbox_dfl, stride, n_bins=16):
        """Decode DFL bbox distribution into (dx, dy, dw, dh).
        
        Args:
            bbox_dfl: (B, 4*n_bins, H, W) — raw logits
            stride: int, grid stride
            n_bins: int, number of bins per offset
        Returns:
            (B, 4, H, W) — (dx, dy, dw, dh) in pixel/log space
        """
        B, C, H, W = bbox_dfl.shape
        pred = bbox_dfl.view(B, 4, n_bins, H, W)
        pred = F.softmax(pred, dim=2)
        bins = torch.linspace(0, 1, n_bins, device=bbox_dfl.device)
        values = (pred * bins.view(1, 1, n_bins, 1, 1)).sum(dim=2)
        values = values * 2.0 - 1.0  # rescale from [0,1] to [-1,1]
        return values


# ──────────────────────────────────────────────
# Full Model: FaceFCNv8
# ──────────────────────────────────────────────

class FaceFCNv8(nn.Module):
    """Full V8 face detector: backbone → BiFPN → 3× independent heads.
    
    Detection levels:
      P3: stride 4,  120×160 grid,  small faces (16-64px)
      P4: stride 8,   60×80 grid,   medium faces (32-128px)
      P5: stride 16,  30×40 grid,   large faces (64-256px)
    """
    strides = {"p3": 4, "p4": 8, "p5": 16}

    def __init__(self, fpn_dim=104, head_dim=104, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.backbone = V8Backbone()
        self.fpn = BiFPN(fpn_dim=fpn_dim)
        self.head_p3 = DetectionHead("p3", in_dim=fpn_dim, hid_dim=head_dim,
                                     obj_bias=-3.0)
        self.head_p4 = DetectionHead("p4", in_dim=fpn_dim, hid_dim=head_dim,
                                     obj_bias=-2.5)
        self.head_p5 = DetectionHead("p5", in_dim=fpn_dim, hid_dim=head_dim,
                                     obj_bias=-2.5)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            c2, c3, c4, c5 = checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False)
            p3_feat, p4_feat, p5_feat = checkpoint.checkpoint(
                self.fpn, c2, c3, c4, c5, use_reentrant=False)
        else:
            c2, c3, c4, c5 = self.backbone(x)
            p3_feat, p4_feat, p5_feat = self.fpn(c2, c3, c4, c5)
        out = {}
        out.update(self.head_p3(p3_feat))
        out.update(self.head_p4(p4_feat))
        out.update(self.head_p5(p5_feat))
        return out

    @staticmethod
    def compute_quality(raw_obj, raw_iou):
        """√(sigmoid(obj) × sigmoid(iou)), quality in [0, 1]."""
        obj_p = torch.sigmoid(raw_obj)
        iou_p = torch.sigmoid(raw_iou)
        return torch.sqrt(obj_p * iou_p + 1e-8)

    @torch.no_grad()
    def detect(self, frame, conf_thresholds=None, nms_iou=0.3):
        """Full detection pipeline: forward → peak-finding → decode → Soft-NMS.
        
        Args:
            frame: BGR numpy array (H, W, 3)
            conf_thresholds: dict of {level: quality_threshold},
                e.g. {"p3": 0.40, "p4": 0.20, "p5": 0.15}
            nms_iou: IoU threshold for Soft-NMS decay
        Returns:
            list of (level, quality, BoundingBox) tuples
        """
        import cv2
        from src.cv.face_tracker import BoundingBox, compute_iou

        if conf_thresholds is None:
            conf_thresholds = {"p3": 0.40, "p4": 0.20, "p5": 0.15}

        device = next(self.parameters()).device
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        out = self.forward(tensor.unsqueeze(0).to(device))

        levels = [
            ("p3", 4, out["p3_obj"][0, 0], out["p3_iou"][0, 0], out["p3_bbox"][0]),
            ("p4", 8, out["p4_obj"][0, 0], out["p4_iou"][0, 0], out["p4_bbox"][0]),
            ("p5", 16, out["p5_obj"][0, 0], out["p5_iou"][0, 0], out["p5_bbox"][0]),
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

            # Decode bbox from DFL distribution
            bbox_tensor = bbox_map.unsqueeze(0)  # add batch dim
            offsets = DetectionHead.decode_bbox(bbox_tensor, stride).squeeze(0)
            offsets = offsets.cpu().numpy()

            ys, xs = np.where(peaks)
            for cy, cx in zip(ys, xs):
                q = float(quality[cy, cx])
                dx = float(offsets[0, cy, cx])
                dy = float(offsets[1, cy, cx])
                dw = float(offsets[2, cy, cx])
                dh = float(offsets[3, cy, cx])

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


# ──────────────────────────────────────────────
# Parameter Verification
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = FaceFCNv8()
    total = sum(p.numel() for p in model.parameters())

    bb = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith('backbone'))
    fp = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith('fpn'))
    h3 = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith('head_p3'))
    h4 = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith('head_p4'))
    h5 = sum(p.numel() for n, p in model.named_parameters()
             if n.startswith('head_p5'))

    print(f"FaceFCNv8 — Independent Heads + BiFPN + DFL")
    print(f"{'='*55}")
    print(f"  Backbone:      {bb:>8,} ({100*bb/total:.0f}%)")
    print(f"  BiFPN:         {fp:>8,} ({100*fp/total:.0f}%)")
    print(f"  Head P3:       {h3:>8,} ({100*h3/total:.0f}%)")
    print(f"  Head P4:       {h4:>8,} ({100*h4/total:.0f}%)")
    print(f"  Head P5:       {h5:>8,} ({100*h5/total:.0f}%)")
    print(f"  Heads total:   {h3+h4+h5:>8,} ({100*(h3+h4+h5)/total:.0f}%)")
    print(f"  {'─'*35}")
    print(f"  Total:         {total:>8,}")
    print(f"  Target:         770,000")
    print(f"  Delta:         {total - 770000:>+8,}")
    print(f"{'='*55}")

    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        out = model(x)
    for k, v in sorted(out.items()):
        print(f"  {k}: {list(v.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params: {total:,}")
