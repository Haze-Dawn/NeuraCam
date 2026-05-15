import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, List

from src.control.kalman import Face, BoundingBox


class FaceFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class FaceCNN:
    def __init__(self, model_path: str = "models/face_cnn.pth",
                 confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.3,
                 input_size: int = 128):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = input_size
        self.scale_factor = 1.15

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceFCN().to(self.device)
        self.model.eval()
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"FaceCNN loaded from {model_path}")
        except Exception as e:
            print(f"FaceCNN load failed ({e}), running in mock mode")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]
        min_size = self.input_size
        detections = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        scale = 1.0
        while min(h, w) * scale >= min_size:
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < min_size or new_h < min_size:
                break

            patch = cv2.resize(rgb, (self.input_size, self.input_size))
            patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
            patch_tensor = patch_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(patch_tensor)
                objness = torch.sigmoid(output[0, 0])
                bbox_offsets = output[0, 1:]

            coord = torch.argmax(objness)
            conf = float(objness.flatten()[coord])
            if conf < self.confidence_threshold:
                scale *= self.scale_factor
                continue

            cy_fmap = coord // objness.shape[1]
            cx_fmap = coord % objness.shape[1]
            fmap_h, fmap_w = objness.shape

            cx_offset = float(bbox_offsets[0].flatten()[coord])
            cy_offset = float(bbox_offsets[1].flatten()[coord])
            log_size = float(bbox_offsets[2].flatten()[coord])

            cell_w = new_w / fmap_w
            cell_h = new_h / fmap_h
            center_x = (cx_fmap + 0.5 + cx_offset) * cell_w / scale
            center_y = (cy_fmap + 0.5 + cy_offset) * cell_h / scale
            size = np.exp(log_size) * self.input_size / scale
            box_w = size
            box_h = size

            x1 = int(center_x - box_w / 2)
            y1 = int(center_y - box_h / 2)
            detections.append(Face(
                bbox=BoundingBox(x=max(0, x1), y=max(0, y1),
                                 w=max(10, int(box_w)), h=max(10, int(box_h))),
                confidence=conf
            ))

            scale *= self.scale_factor

        detections = self._nms(detections)
        return detections

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
