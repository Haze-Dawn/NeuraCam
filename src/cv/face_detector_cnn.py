import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List
from src.cv.face_tracker import Face, BoundingBox, compute_iou


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
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Conv2d(64, 128, kernel_size=1)
        self.fuse_conv = nn.Conv2d(256, 128, kernel_size=1)
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
        skip = self.skip_conv(x)
        x = self.block4(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse_conv(x)
        x = self.head(x)
        return x


class FaceCNN:
    def __init__(self, model_path: str = "models/face_cnn.pth",
                 confidence_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.25,
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
        base_model = FaceFCN()
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            base_model.load_state_dict(state_dict)
            base_model.eval()
            try:
                self.model = torch.jit.script(base_model)
                print(f"FaceCNN loaded from {model_path} (TorchScript optimized)")
            except Exception as js_err:
                self.model = base_model
                print(f"FaceCNN loaded from {model_path} "
                      f"(TorchScript unavailable: {js_err})")
            self.model.to(self.device)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FaceCNN model not found at {model_path}. "
                "Training is required before running detection."
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FaceCNN model from {model_path}: {e}. "
                "Ensure the model file is valid and compatible."
            ) from e

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]
        min_size = self.input_size

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Descending scale pyramid: 1.0 → 0.87 → 0.76 → 0.66 → 0.57
        # Each step divides by 1.15 to detect progressively smaller faces
        n_scales = 5
        scales = [1.0 / (self.scale_factor ** i) for i in range(n_scales)]
        best_conf = 0.0

        # Per-scale detection lists for score averaging across scales
        scale_detections: List[List[Face]] = [[] for _ in scales]

        for si, scale in enumerate(scales):
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

                if bw > 5 and bh > 5 and 20 <= box_size <= 500:
                    if conf > best_conf:
                        best_conf = conf
                    scale_detections[si].append(Face(
                        bbox=BoundingBox(x=x1, y=y1, w=bw, h=bh),
                        confidence=conf
                    ))

        # Score averaging across scales: group detections that overlap across
        # different scales of the pyramid. The same face at adjacent scales
        # produces similar bboxes with slightly different confidences.
        # Averaging them reduces noise and produces smoother tracking.
        all_detections = []
        for sd in scale_detections:
            all_detections.extend(sd)

        if not all_detections:
            return []

        # NMS with score averaging: group overlapping detections,
        # replace each group with the highest-confidence bbox at average confidence
        all_detections = sorted(all_detections, key=lambda d: d.confidence, reverse=True)
        groups = []
        assigned = [False] * len(all_detections)
        for i, det in enumerate(all_detections):
            if assigned[i]:
                continue
            group = [det]
            assigned[i] = True
            for j in range(i + 1, len(all_detections)):
                if assigned[j]:
                    continue
                iou = compute_iou(det.bbox, all_detections[j].bbox)
                if iou > self.nms_iou_threshold:
                    group.append(all_detections[j])
                    assigned[j] = True
            groups.append(group)

        merged = []
        for group in groups:
            avg_conf = float(np.mean([d.confidence for d in group]))
            best = max(group, key=lambda d: d.confidence)
            merged.append(Face(
                bbox=BoundingBox(x=best.bbox.x, y=best.bbox.y,
                                 w=best.bbox.w, h=best.bbox.h),
                confidence=avg_conf
            ))

        merged.sort(key=lambda d: d.confidence, reverse=True)
        return self._filter_by_confidence_ratio(merged)

    def _filter_by_confidence_ratio(self, detections: List[Face]) -> List[Face]:
        if not detections:
            return []
        max_conf = max(d.confidence for d in detections)
        if max_conf >= 0.5:
            ratio_threshold = 0.3 * max_conf
            kept = []
            for d in detections:
                if d.confidence >= ratio_threshold:
                    kept.append(d)
                else:
                    # Exempt detections with low IoU against ALL kept detections.
                    # A detection with IoU < 0.1 vs every already-kept detection
                    # is likely a different face rather than a duplicate.
                    is_different_face = True
                    for k in kept:
                        if compute_iou(d.bbox, k.bbox) >= 0.1:
                            is_different_face = False
                            break
                    if is_different_face:
                        kept.append(d)
            return kept
        return detections


