"""
FaceCNN V0 — Lightweight face detector for embedded deployment.
Architecture: Depthwise separable conv backbone + feature pyramid + multi-scale heads.

Designed for WIDER Face training at 320x320 input, 3 detection levels (strides 8/16/32).

Parameter budget: ~187K
Design: 11 separable conv blocks (4 MaxPool stages), 3-level FPN, direct cls/obj/bbox heads.

Input:  320x320 BGR, pixel range [0, 255]
Output: dict with cls/obj/bbox at strides 8, 16, 32 (raw logits, no sigmoid applied)
"""
import math, numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv(nn.Module):
    """Depthwise separable conv: 1x1 proj -> 3x3 depthwise -> BN -> ReLU."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1, bias=True)
        self.depth = nn.Conv2d(c_out, c_out, 3, 1, 1, groups=c_out, bias=True)
        self.norm = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.depth(self.proj(x))))


class HeadConv(nn.Module):
    """Detection head: 1x1 proj -> 3x3 depthwise (no BN, no activation)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1, bias=True)
        self.depth = nn.Conv2d(c_out, c_out, 3, 1, 1, groups=c_out, bias=True)

    def forward(self, x):
        return self.depth(self.proj(x))


class FaceCNNV0(nn.Module):
    """187K face detector. 11 depthwise separable conv blocks, 3-level FPN, direct heads."""
    def __init__(self):
        super().__init__()
        C = 128  # peak channels

        # Entry stem (stride 2)
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=True),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        # Encoder blocks (11 total, 4 MaxPool stages)
        # Stage 1: stride 2 -> 4 (1 block + pool)
        self.s1 = nn.Sequential(
            SeparableConv(32, 32),
            nn.MaxPool2d(2),
        )

        # Stage 2: stride 4 (2 blocks, no pool)
        self.s2 = nn.Sequential(
            SeparableConv(32, 32),
            SeparableConv(32, 64),
        )

        # Stage 3: stride 4 (2 blocks, no pool)
        self.s3 = nn.Sequential(
            SeparableConv(64, 64),
            SeparableConv(64, C),
        )

        # Stage 4: stride 8 (2 blocks, tap before next pool)
        self.s4 = nn.Sequential(
            SeparableConv(C, C),
            SeparableConv(C, C),
        )

        # Stage 5: stride 16 (2 blocks, tap before next pool)
        self.s5 = nn.Sequential(
            SeparableConv(C, C),
            SeparableConv(C, C),
        )

        # Stage 6: stride 32 (2 blocks, tap)
        self.s6 = nn.Sequential(
            SeparableConv(C, C),
            SeparableConv(C, C),
        )

        # FPN laterals
        self.fpn_hi = SeparableConv(C, C)  # stride 32 level
        self.fpn_md = SeparableConv(C, C)  # stride 16 level
        self.fpn_lo = SeparableConv(C, C)  # stride 8 level

        # Detection heads (3 scales x cls/obj/bbox)
        self.cls_8 = HeadConv(C, 1)
        self.obj_8 = HeadConv(C, 1)
        self.bbox_8 = HeadConv(C, 4)
        self.cls_16 = HeadConv(C, 1)
        self.obj_16 = HeadConv(C, 1)
        self.bbox_16 = HeadConv(C, 4)
        self.cls_32 = HeadConv(C, 1)
        self.obj_32 = HeadConv(C, 1)
        self.bbox_32 = HeadConv(C, 4)

    def forward(self, x):
        # Encoder
        x = self.entry(x)    # stride 2
        x = self.s1(x)       # stride 4
        x = self.s2(x)       # stride 4 (2 blocks)
        x = self.s3(x)       # stride 4 (2 blocks)

        x = F.max_pool2d(x, 2)  # stride 8
        x = self.s4(x)       # stride 8 (2 blocks)
        f_lo = x             # stride 8 tap

        x = F.max_pool2d(x, 2)  # stride 16
        x = self.s5(x)       # stride 16 (2 blocks)
        f_md = x             # stride 16 tap

        x = F.max_pool2d(x, 2)  # stride 32
        x = self.s6(x)       # stride 32 (2 blocks)
        f_hi = x             # stride 32 tap

        # FPN
        p_hi = self.fpn_hi(f_hi)
        p_md = self.fpn_md(f_md + F.interpolate(p_hi, scale_factor=2, mode='nearest'))
        p_lo = self.fpn_lo(f_lo + F.interpolate(p_md, scale_factor=2, mode='nearest'))

        # Heads
        return {
            'cls_8': self.cls_8(p_lo), 'obj_8': self.obj_8(p_lo), 'bbox_8': self.bbox_8(p_lo),
            'cls_16': self.cls_16(p_md), 'obj_16': self.obj_16(p_md), 'bbox_16': self.bbox_16(p_md),
            'cls_32': self.cls_32(p_hi), 'obj_32': self.obj_32(p_hi), 'bbox_32': self.bbox_32(p_hi),
        }

    @torch.no_grad()
    def detect(self, frame, conf_threshold=0.5, nms_iou=0.45, input_size=640):
        h, w = frame.shape[:2]
        isz = input_size
        img = cv2.resize(frame, (isz, isz))
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        t = t.to(next(self.parameters()).device)
        out = self.forward(t)

        all_dets = []
        for lname, stride in [('8', 8), ('16', 16), ('32', 32)]:
            cls_p = torch.sigmoid(out[f'cls_{lname}'][0, 0]).cpu().numpy()
            obj_p = torch.sigmoid(out[f'obj_{lname}'][0, 0]).cpu().numpy()
            score = cls_p * obj_p
            bbox_map = out[f'bbox_{lname}'][0].cpu().numpy()

            sy, sx = h / isz, w / isz
            rows, cols = np.where(score > conf_threshold)
            for cy, cx in zip(rows, cols):
                q = float(score[cy, cx])
                dx = float(bbox_map[0, cy, cx])
                dy = float(bbox_map[1, cy, cx])
                dw = float(bbox_map[2, cy, cx])
                dh = float(bbox_map[3, cy, cx])

                box_cx = (cx + 0.5 + dx) * stride
                box_cy = (cy + 0.5 + dy) * stride
                box_w = float(np.exp(np.clip(dw, -2, 5))) * stride
                box_h = float(np.exp(np.clip(dh, -2, 5))) * stride

                if box_w < 3 or box_h < 3:
                    continue
                x1 = int((box_cx - box_w / 2) * sx)
                y1 = int((box_cy - box_h / 2) * sy)
                bw = int(box_w * sx)
                bh = int(box_h * sy)
                all_dets.append((q, [x1, y1, bw, bh]))

        all_dets.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for det in all_dets:
            md = max((self._iou(det[1], k[1]) for k in kept), default=0.0)
            if md <= nms_iou:
                kept.append(det)
        return kept

    @staticmethod
    def _iou(a, b):
        xo = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
        yo = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
        inter = xo * yo
        return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-8) if inter > 0 else 0.0


if __name__ == "__main__":
    m = FaceCNNV0()
    n = sum(p.numel() for p in m.parameters())
    print(f"FaceCNN V0: {n:,} params")
    t = torch.randn(1, 3, 320, 320)
    out = m(t)
    for k, v in out.items():
        print(f"  {k}: {list(v.shape)}")
