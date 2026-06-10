import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, inplace=self.inplace) / 6.0


class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                    padding=dilation, dilation=dilation,
                                    groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = HardSwish(inplace=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = HardSwish(inplace=False)

    def forward(self, x):
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(4, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=False),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x)


class V7_1Backbone(nn.Module):
    """19 DSConv blocks, HardSwish throughout, max 128ch.
    S4 removed (was dead — BN gamma frozen at 1.0 across all channels).
    Outputs C2 (48ch, stride 4), C3 (96ch, stride 8), C4 (128ch, stride 16)."""
    def __init__(self, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            HardSwish(inplace=False),
        )
        self.s1 = nn.Sequential(
            DSConvBlock(32, 48, stride=2),
            DSConvBlock(48, 48),
        )
        self.s2_down = DSConvBlock(48, 96, stride=2)
        self.s2 = nn.Sequential(*[DSConvBlock(96, 96) for _ in range(8)])
        self.s3_down = DSConvBlock(96, 128, stride=2)
        self.s3 = nn.Sequential(*[DSConvBlock(128, 128) for _ in range(6)])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_checkpointing and self.training:
            return self._forward_checkpointed(x)
        return self._forward_direct(x)

    def _forward_direct(self, x):
        c1 = self.stem(x)
        c2 = self.s1(c1)
        x = self.s2_down(c2)
        c3 = self.s2(x)
        x = self.s3_down(c3)
        c4 = self.s3(x)
        return c1, c2, c3, c4

    def _forward_checkpointed(self, x):
        c1 = checkpoint(self.stem, x, use_reentrant=False)
        c2 = checkpoint(self.s1, c1, use_reentrant=False)
        x = checkpoint(self.s2_down, c2, use_reentrant=False)
        c3 = checkpoint(self.s2, x, use_reentrant=False)
        x = checkpoint(self.s3_down, c3, use_reentrant=False)
        c4 = checkpoint(self.s3, x, use_reentrant=False)
        return c1, c2, c3, c4


class WeightedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, channels))
    def forward(self, feat_a, feat_b):
        w = F.softmax(self.weights, dim=0)
        return w[0:1, :, None, None] * feat_a + w[1:2, :, None, None] * feat_b


class V7_1FPN(nn.Module):
    """128ch FPN with C2 fusion. Removed lat5/w_fuse_p4/se5 (dead from S4 deletion).
    New forward: C4→lat4→refine4→se4→P4→upsample→fuse→C3→refine3→se3→P3→upsample→fuse→C2→refine2."""
    def __init__(self, fpn_dim=128):
        super().__init__()
        self.lat2 = nn.Conv2d(48, fpn_dim, 1, bias=False)  # C2 → P4 level
        self.lat3 = nn.Conv2d(96, fpn_dim, 1, bias=False)
        self.lat4 = nn.Conv2d(128, fpn_dim, 1, bias=False)
        self.w_fuse_p3 = WeightedFusion(fpn_dim)
        self.refine2 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            HardSwish(inplace=False),
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            HardSwish(inplace=False),
        )
        self.refine4 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            HardSwish(inplace=False),
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

    def forward(self, c1, c2, c3, c4):
        p4 = self.lat4(c4)
        p4 = self.refine4(p4)
        p4 = self.se4(p4) + p4

        p3 = self.w_fuse_p3(
            self.lat3(c3),
            F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        )
        p3 = self.refine3(p3)
        p3 = self.se3(p3) + p3

        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        c2_proj = self.lat2(c2)
        p4_feat = self.refine2(c2_proj + p3_up)

        return p4_feat, c1  # c1 (stride 2) goes to P2 head


class V7_1P2Head(nn.Module):
    """Lightweight stride-2 head for small faces (10-25px).
    Grid: 320x400 at 640x800 input = 128,000 cells."""
    def __init__(self, in_dim=32, hid_dim=64, obj_bias=-2.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hid_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=False),
        )
        self.obj = nn.Conv2d(hid_dim, 1, 1)
        self.iou = nn.Conv2d(hid_dim, 1, 1)
        self.bbox = nn.Conv2d(hid_dim, 4, 1)
        self._init_weights(obj_bias)

    def _init_weights(self, obj_bias):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.obj.bias, obj_bias)
        nn.init.zeros_(self.bbox.bias)
        self.bbox.bias.data[2:] = -1.0

    def forward(self, x):
        x = self.conv1(x)
        return {
            "p2_obj": self.obj(x),
            "p2_iou": self.iou(x),
            "p2_bbox": self.bbox(x),
        }


class V7_1Head(nn.Module):
    """5-layer P4 detection head with direct bbox regression."""
    def __init__(self, in_dim=128, hid_dim=192, mid_dim=128, pred_dim=96,
                 obj_bias=-2.5, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=False),
        )
        self.conv2_dw = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, 3, padding=1, groups=hid_dim, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=False),
        )
        self.conv2_pw = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            HardSwish(inplace=False),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hid_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            HardSwish(inplace=False),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(mid_dim, pred_dim, 1, bias=False),
            nn.BatchNorm2d(pred_dim),
            HardSwish(inplace=False),
        )
        self.obj = nn.Conv2d(pred_dim, 1, 1)
        self.iou = nn.Conv2d(pred_dim, 1, 1)
        self.bbox = nn.Conv2d(pred_dim, 4, 1)
        self._init_weights(obj_bias)

    def _init_weights(self, obj_bias):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.obj.bias, obj_bias)
        nn.init.zeros_(self.iou.bias)
        nn.init.zeros_(self.bbox.bias)
        self.bbox.bias.data[2:] = -1.0

    def forward(self, feat):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, feat, use_reentrant=False)
        return self._forward_impl(feat)

    def _forward_impl(self, feat):
        x = self.conv1(feat)
        x = self.conv2_dw(x)
        x = self.conv2_pw(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return {
            "obj": self.obj(x),
            "iou": self.iou(x),
            "bbox": self.bbox(x),
        }


class FaceFCNv7_1(nn.Module):
    """V7.1 — Dual-level (P2+P4) face detector with C2 fusion.
    Architecture:
      V7_1Backbone → V7_1FPN (P4 level with C2 fusion)
                  → V7_1P2Head (stride 2, small faces)
                  → V7_1Head (stride 4, medium-large faces)

    P4 head: C2 fusion + interpolated P3 → 5-layer DW head → obj+iou+bbox
    P2 head: Direct C2 features → 2-layer head → obj+iou+bbox

    Inference: cross-level NMS + multi-scale support.
    Targets: Beat SCRFD-0.5GF on WIDER Face Easy/Medium/Hard.
    """
    strides = {"p2": 2, "p4": 4}

    def __init__(self, fpn_dim=128, head_in_dim=128, head_hid=192,
                 head_mid=128, head_pred=96, obj_bias=-2.5,
                 use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.backbone = V7_1Backbone(use_checkpointing=use_checkpointing)
        self.fpn = V7_1FPN(fpn_dim=fpn_dim)
        self.head = V7_1Head(
            in_dim=head_in_dim, hid_dim=head_hid,
            mid_dim=head_mid, pred_dim=head_pred, obj_bias=obj_bias,
            use_checkpointing=use_checkpointing,
        )
        self.p2_head = V7_1P2Head(in_dim=32, hid_dim=64, obj_bias=obj_bias)

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        p4_feat, c1_raw = self.fpn(c1, c2, c3, c4)

        if self.training and self.use_checkpointing:
            out_p4 = checkpoint(self.head, p4_feat, use_reentrant=False)
            out_p2 = self.p2_head(c1_raw)  # P2 is lightweight, no checkpoint needed
        else:
            out_p4 = self.head(p4_feat)
            out_p2 = self.p2_head(c1_raw)

        return {
            "obj": out_p4["obj"], "iou": out_p4["iou"], "bbox": out_p4["bbox"],
            "p2_obj": out_p2["p2_obj"], "p2_iou": out_p2["p2_iou"],
            "p2_bbox": out_p2["p2_bbox"],
        }

    @staticmethod
    def compute_quality(obj_logits, iou_logits):
        obj_p = torch.sigmoid(obj_logits)
        iou_p = torch.sigmoid(iou_logits)
        return torch.sqrt(obj_p * iou_p + 1e-8)

    @torch.no_grad()
    def detect(self, frame, conf_thresholds=None, nms_iou=0.3, tta=False, scales=None):
        """Full detection pipeline: multi-scale -> peak-finding -> decode -> Soft-NMS.
        
        Uses quality = sqrt(sigmoid(obj) * sigmoid(iou)) for ranking.
        Supports multi-scale inference with scale skipping.
        
        Args:
            frame: BGR numpy array (H, W, 3)
            conf_thresholds: dict of per-level thresholds:
                {'p4': 0.25, 'p2': 0.15}. If None, uses 0.25 for all levels.
            nms_iou: IoU threshold for Soft-NMS (default 0.3)
            tta: if True, run flip averaging (2x forward passes)
            scales: list of scale factors, e.g. [1.0, 1.5, 2.0].
                    None = [1.0] (tracking mode for speed).
        Returns:
            list of (quality, BoundingBox) tuples
        """
        import cv2
        from src.cv.face_tracker import BoundingBox

        if tta:
            return self._detect_tta(frame, conf_thresholds, nms_iou)

        if conf_thresholds is None:
            conf_thresholds = {"p4": 0.25, "p2": 0.15}
        if scales is None:
            scales = [1.0]

        device = next(self.parameters()).device
        orig_h, orig_w = frame.shape[:2]
        all_dets = []
        bbox_size_thresh = 5

        for sf in scales:
            if sf != 1.0:
                h = int(round(orig_h * sf))
                w = int(round(orig_w * sf))
                scaled = cv2.resize(frame, (w, h))
                inv_sf = 1.0 / sf
            else:
                scaled = frame
                h, w = orig_h, orig_w
                inv_sf = 1.0

            rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            out = self.forward(tensor.unsqueeze(0).to(device))

            # Skip remaining scales if we found detections at this scale
            # (only applies when scales has multiple entries — scale skipping)
            found_at_this_scale = False

            for level, stride, prefix in [("p4", 4, ""), ("p2", 2, "p2_")]:
                thresh = conf_thresholds.get(level, 0.25)
                obj_map = out[f"{prefix}obj"][0, 0].cpu().numpy()
                iou_map = out[f"{prefix}iou"][0, 0].cpu().numpy()
                bbox_map = out[f"{prefix}bbox"][0].cpu().numpy()
                quality = np.sqrt(obj_map * iou_map + 1e-8)

                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(quality, kernel)
                peaks = (quality == dilated) & (quality > thresh)

                if peaks.any():
                    found_at_this_scale = True
                    ys, xs = np.where(peaks)
                    for cy, cx in zip(ys, xs):
                        q = float(quality[cy, cx])
                        dx = float(bbox_map[0, cy, cx])
                        dy = float(bbox_map[1, cy, cx])
                        dw = float(np.clip(bbox_map[2, cy, cx], -2, 5))
                        dh = float(np.clip(bbox_map[3, cy, cx], -2, 5))

                        box_cx = (cx + 0.5 + dx) * stride * inv_sf
                        box_cy = (cy + 0.5 + dy) * stride * inv_sf
                        box_w = float(np.exp(dw)) * stride * inv_sf
                        box_h = float(np.exp(dh)) * stride * inv_sf

                        if box_w < bbox_size_thresh or box_h < bbox_size_thresh:
                            continue
                        x1 = int(max(0, box_cx - box_w / 2))
                        y1 = int(max(0, box_cy - box_h / 2))
                        bw = int(min(box_w, orig_w - x1))
                        bh = int(min(box_h, orig_h - y1))
                        if bw < bbox_size_thresh or bh < bbox_size_thresh:
                            continue
                        all_dets.append(
                            (q, BoundingBox(x=x1, y=y1, w=bw, h=bh))
                        )

            # Scale skipping: if tracking and we found detections, stop
            if found_at_this_scale and len(scales) > 1:
                break

        return self._soft_nms(all_dets, nms_iou)

    def _detect_tta(self, frame, conf_thresholds, nms_iou):
        import cv2
        from src.cv.face_tracker import compute_iou, BoundingBox
        dets1 = self.detect(frame, conf_thresholds=conf_thresholds, nms_iou=nms_iou, tta=False)
        flipped = cv2.flip(frame, 1)
        dets2 = self.detect(flipped, conf_thresholds=conf_thresholds, nms_iou=nms_iou, tta=False)
        h, w = frame.shape[:2]
        dets2 = [(q, BoundingBox(x=w - b.x - b.w, y=b.y, w=b.w, h=b.h))
                 for q, b in dets2]
        merged = dets1 + dets2
        merged.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for det in merged:
            keep = True
            for k in kept:
                if compute_iou(det[1], k[1]) > nms_iou:
                    keep = False
                    break
            if keep:
                kept.append(det)
        return kept

    def _soft_nms(self, dets, nms_iou=0.3):
        from src.cv.face_tracker import compute_iou
        dets.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for det in dets:
            max_decay = 0.0
            for k in kept:
                iou = compute_iou(det[1], k[1])
                if iou > nms_iou:
                    c1, c2 = det[1], k[1]
                    center_dist = (c1.center_x - c2.center_x)**2 + (c1.center_y - c2.center_y)**2
                    diag = max(c1.w, c2.w)**2 + max(c1.h, c2.h)**2
                    diou = iou - center_dist / (diag + 1e-8)
                    if diou > 0:
                        max_decay = max(max_decay, diou)
            if max_decay > 0:
                decayed = det[0] * (1.0 - max_decay)
                if decayed > 0.01:
                    kept.append((decayed, det[1]))
            else:
                kept.append(det)
        return kept

    def sanity_check(self):
        device = next(self.parameters()).device
        self.eval()
        collapse_threshold = 1e-5
        warning_threshold = 0.10
        with torch.no_grad():
            zero_in = torch.zeros(1, 3, 480, 640, device=device)
            out_zero = self.forward(zero_in)
            for k in ["obj", "p2_obj"]:
                v = out_zero[k]
                if not torch.isfinite(v).all():
                    nans = torch.isnan(v).sum().item()
                    infs = torch.isinf(v).sum().item()
                    raise RuntimeError(f"Model sanity FAILED: {k} contains NaN={nans}, Inf={infs}")
                std = v.std().item()
                if std < collapse_threshold:
                    import warnings as warn
                    warn.warn(f"Model sanity WARNING: {k} std={std:.2e} (collapsed — expected for untrained model)",
                              category=UserWarning)
            one_in = torch.ones(1, 3, 480, 640, device=device)
            out_one = self.forward(one_in)
            for k in ["obj", "p2_obj"]:
                diff = (torch.sigmoid(out_one[k]) - torch.sigmoid(out_zero[k])).abs().max().item()
                if diff < collapse_threshold:
                    import warnings as warn
                    warn.warn(f"Model sanity WARNING: {k} content delta={diff:.2e} (expected for untrained model)",
                              category=UserWarning)
            max_sig = max(torch.sigmoid(out_zero["obj"]).max().item(),
                          torch.sigmoid(out_one["obj"]).max().item(),
                          torch.sigmoid(out_zero["p2_obj"]).max().item())
            if max_sig < warning_threshold:
                import warnings as warn
                warn.warn(f"Model sanity WARNING: max sigmoid={max_sig:.4f} (below {warning_threshold:.2f})",
                          category=UserWarning)
            print("Model sanity check:")
            print(f"  NaN/Inf:         {'PASS' if all(torch.isfinite(out_zero[k]).all() for k in ['obj','p2_obj']) else 'FAIL'}")
            for k in ["obj", "p2_obj"]:
                std = out_zero[k].std().item()
                m = torch.sigmoid(out_zero[k]).max().item()
                print(f"  {k:6s}: std={std:.2e} max_sig={m:.4f}")


if __name__ == "__main__":
    model = FaceFCNv7_1()
    total = sum(p.numel() for p in model.parameters())
    bb = sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n)
    fp = sum(p.numel() for n, p in model.named_parameters() if 'fpn' in n)
    hd = sum(p.numel() for n, p in model.named_parameters() if 'head' in n)
    p2_hd = sum(p.numel() for n, p in model.named_parameters() if 'p2_head' in n)
    print("FaceFCNv7_1 — Dual-Level (P2+P4) Face Detector with C2 Fusion")
    print("=" * 58)
    print(f"  Backbone:    {bb:>8,} ({100*bb/total:.1f}%)")
    print(f"  FPN:         {fp:>8,} ({100*fp/total:.1f}%)")
    print(f"  P4 Head:     {hd:>8,} ({100*hd/total:.1f}%)")
    print(f"  P2 Head:     {p2_hd:>8,} ({100*p2_hd/total:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total:       {total:>8,} ({total*4/1e6:.1f} MB FP32)")
    print(f"  vs SCRFD-0.5GF (570K): {570000 - total:+,} smaller ({100*(570000-total)/570000:.1f}%)")
    print("=" * 58)
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        out = model(x)
    for k, v in out.items():
        print(f"  {k}: {list(v.shape)}")
