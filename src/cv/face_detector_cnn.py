import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List
from src.control.kalman import Face, BoundingBox


class FaceFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, 4, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head(x)
        return x


class FaceCNN:
    def __init__(self, model_path: str = "models/face_cnn.pth",
                 confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.3,
                 input_size: int = 128,
                 skip_scale_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = input_size
        self.skip_scale_threshold = skip_scale_threshold
        self.scale_factor = 1.15
        self.stride = 8
        self.grid_cells = input_size // self.stride

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceFCN().to(self.device)
        self.model.eval()
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"FaceCNN loaded from {model_path}")
        except Exception as e:
            print(f"FaceCNN load failed ({e}), running with random weights")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]
        min_size = self.input_size
        detections = []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Descending scale pyramid: 1.0 → 0.87 → 0.76 → 0.66 → 0.57
        # Each step divides by 1.15 to detect progressively smaller faces
        n_scales = 5
        scales = [1.0 / (self.scale_factor ** i) for i in range(n_scales)]
        best_conf = 0.0  # track max confidence across scales for early exit

        for scale in scales:
            # Confidence-based scale skipping: if we already have a high-confidence
            # detection, skip smaller scales (face is already found at this size)
            if best_conf >= self.skip_scale_threshold:
                continue
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < min_size or new_h < min_size:
                continue

            scaled = cv2.resize(rgb, (new_w, new_h))
            pad_h = (self.stride - new_h % self.stride) % self.stride
            pad_w = (self.stride - new_w % self.stride) % self.stride
            if pad_h > 0 or pad_w > 0:
                scaled = cv2.copyMakeBorder(scaled, 0, pad_h, 0, pad_w,
                                            cv2.BORDER_REFLECT)

            tensor = torch.from_numpy(scaled).float().permute(2, 0, 1) / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                out = output[0].cpu().numpy()

            objness = 1.0 / (1.0 + np.exp(-out[0]))
            cells_y, cells_x = np.where(objness > self.confidence_threshold)

            for cy, cx in zip(cells_y, cells_x):
                conf = float(objness[cy, cx])
                dx = float(out[1, cy, cx])
                dy = float(out[2, cy, cx])
                log_size = float(out[3, cy, cx])

                center_x = (cx + 0.5 + dx) * self.stride / scale
                center_y = (cy + 0.5 + dy) * self.stride / scale
                size = np.exp(log_size) * self.stride / scale
                box_size = max(size, 10.0)

                x1 = int(center_x - box_size / 2)
                y1 = int(center_y - box_size / 2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                bw = min(int(box_size), w - x1)
                bh = min(int(box_size), h - y1)

                if bw > 5 and bh > 5:
                    if conf > best_conf:
                        best_conf = conf
                    detections.append(Face(
                        bbox=BoundingBox(x=x1, y=y1, w=bw, h=bh),
                        confidence=conf
                    ))

        return self._nms(detections)

    def _nms(self, detections: List[Face]) -> List[Face]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        for det in detections:
            keep_flag = True
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou > self.nms_iou_threshold and det.confidence < kept.confidence:
                    keep_flag = False
                    break
            if keep_flag:
                keep.append(det)
        return keep

    @staticmethod
    def _compute_iou(a: BoundingBox, b: BoundingBox) -> float:
        x_overlap = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        y_overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        intersection = x_overlap * y_overlap
        if intersection == 0:
            return 0.0
        area_a = a.w * a.h
        area_b = b.w * b.h
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0
