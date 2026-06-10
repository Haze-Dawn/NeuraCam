"""
FaceVerifierV1 — Tiny verification CNN for cascade face detection.
Architecture: 6 depthwise-separable conv blocks → GAP → 2 FC
Params:      ~8,697
Input:       (B, 3, 32, 32) RGB [0, 1]
Output:      (B, 1) logits (apply sigmoid for face probability)
"""

import math
import torch
import torch.nn as nn
import numpy as np
import cv2


class DSConv(nn.Module):
    """Depthwise-separable conv block: DW → PW → BN → ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.relu(self.bn1(x))
        x = self.pw(x)
        x = self.relu(self.bn2(x))
        return x


class FaceVerifierV1(nn.Module):
    """Tiny verification CNN (<10K params)."""
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size = input_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DSConv(8, 16, stride=1),
            DSConv(16, 24, stride=2),
            DSConv(24, 32, stride=1),
            DSConv(32, 48, stride=2),
            DSConv(48, 64, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    @torch.no_grad()
    def predict_proba(self, crops):
        """Batch predict face probabilities from numpy RGB crops."""
        t = torch.from_numpy(crops).float().permute(0, 3, 1, 2) / 255.0
        return torch.sigmoid(self.forward(t)).squeeze(-1).cpu().numpy()


class FaceVerifier:
    """Wraps FaceVerifierV1: extract crops, run inference, filter detections."""
    def __init__(self, model, input_size=32, conf_threshold=0.5, max_proposals=200):
        self.model = model
        self.model.eval()
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.max_proposals = max_proposals

    @torch.no_grad()
    def filter(self, detections, frame):
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x[0], reverse=True)[:self.max_proposals]
        crops, valid = [], []
        for i, (score, box) in enumerate(detections):
            x1, y1 = int(max(0, box.x)), int(max(0, box.y))
            x2 = int(min(frame.shape[1], box.x + box.w))
            y2 = int(min(frame.shape[0], box.y + box.h))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = cv2.resize(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
                              (self.input_size, self.input_size))
            crops.append(crop); valid.append(i)
        if not crops:
            return []
        v_scores = self.model.predict_proba(np.stack(crops).astype(np.float32))
        results = []
        for idx, v_score in zip(valid, v_scores):
            if v_score >= self.conf_threshold:
                combined = math.sqrt(detections[idx][0] * v_score)
                results.append((combined, detections[idx][1]))
        results.sort(key=lambda x: x[0], reverse=True)
        return self._soft_nms(results)

    def _soft_nms(self, dets, iou_thresh=0.3):
        kept = []
        for det in dets:
            md = 0.0
            for k in kept:
                iou = self._iou(det[1], k[1])
                if iou > iou_thresh:
                    md = max(md, iou)
            if md > 0:
                d = det[0] * (1.0 - md)
                if d > 0.01:
                    kept.append((d, det[1]))
            else:
                kept.append(det)
        return kept

    @staticmethod
    def _iou(a, b):
        xo = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        yo = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        inter = xo * yo
        if inter == 0:
            return 0.0
        u = a.w * a.h + b.w * b.h - inter
        return inter / u if u > 0 else 0.0
