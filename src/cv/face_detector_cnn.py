import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from src.cv.face_tracker import Face, BoundingBox, compute_iou
from src.cv.face_detector_v71 import FaceFCNv7_1 as FaceFCNv7_1Model


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    padding=padding, dilation=dilation,
                                    groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# FaceFCN v4.0 — Original Architecture (kept for backward compatibility)
# ============================================================================

class FaceFCN(nn.Module):
    def __init__(self, num_anchors=3, use_depthwise=True, use_se=True):
        super().__init__()
        self.num_anchors = num_anchors
        self.use_se = use_se

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

        if use_depthwise:
            self.block3 = nn.Sequential(
                DepthwiseSeparableConv(32, 64, 3, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.block4 = nn.Sequential(
                DepthwiseSeparableConv(64, 128, 3, 2, dilation=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
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

        self.se = SEBlock(128) if use_se else nn.Identity()

        self.head_obj = nn.Conv2d(128, 1, kernel_size=1)
        self.head_bbox = nn.Conv2d(128, num_anchors * 3, kernel_size=1)

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
        x = self.se(x)
        obj = self.head_obj(x)
        bbox = self.head_bbox(x)
        return torch.cat([obj, bbox], dim=1)


ANCHOR_SCALES = [1.5, 3.0, 6.0]


class FaceCNN:
    def __init__(self, model_path: str = "models/face_cnn.pth",
                 confidence_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.25,
                 input_size: int = 128,
                 skip_scale_threshold: float = 0.9,
                 num_anchors: int = 3,
                 use_depthwise: bool = True,
                 use_se: bool = True):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = input_size
        self.skip_scale_threshold = skip_scale_threshold
        self.scale_factor = 1.15
        self.stride = 8
        self.grid_cells = input_size // self.stride
        self.num_anchors = num_anchors
        self.anchor_scales = ANCHOR_SCALES[:num_anchors]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = FaceFCN(num_anchors=num_anchors,
                             use_depthwise=use_depthwise,
                             use_se=use_se)
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            has_ds = any(k.startswith("block3.0.depthwise") or k.startswith("block4.0.depthwise")
                        for k in state_dict.keys())
            if not has_ds:
                old_keys = [k for k in state_dict.keys()
                           if k.startswith("block3.0.") or k.startswith("block4.0.")]
                if old_keys:
                    for k in old_keys:
                        new_k = k.replace(".0.", ".0.depthwise.")
                        state_dict[new_k] = state_dict.pop(k)

            if "head.weight" in state_dict:
                old_w = state_dict.pop("head.weight")
                old_b = state_dict.pop("head.bias", None)
                if old_w.shape[0] == 10:
                    state_dict["head_obj.weight"] = old_w[0:1]
                    state_dict["head_bbox.weight"] = old_w[1:]
                    if old_b is not None:
                        state_dict["head_obj.bias"] = old_b[0:1]
                        state_dict["head_bbox.bias"] = old_b[1:]
                elif old_w.shape[0] == 4:
                    state_dict["head_obj.weight"] = old_w[0:1]
                    new_bb = old_w[1:4].repeat(num_anchors, 1, 1, 1)
                    state_dict["head_bbox.weight"] = new_bb
                    if old_b is not None:
                        state_dict["head_obj.bias"] = old_b[0:1]
                        new_bb_b = old_b[1:4].repeat(num_anchors)
                        state_dict["head_bbox.bias"] = new_bb_b

            base_model.load_state_dict(state_dict, strict=False)
            base_model.eval()
            try:
                self.model = torch.jit.script(base_model)
                print(f"FaceCNN loaded from {model_path} (TorchScript optimized, "
                      f"{num_anchors} anchors, depthwise={use_depthwise})")
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

        n_scales = 5
        scales = [1.0 / (self.scale_factor ** i) for i in range(n_scales)]
        best_conf = 0.0

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

            raw_h, raw_w = scaled.shape[:2]
            actual_grid_h = raw_h // self.stride
            actual_grid_w = raw_w // self.stride

            for cy, cx in zip(cells_y, cells_x):
                if cy >= actual_grid_h or cx >= actual_grid_w:
                    continue
                conf = float(objness[cy, cx])

                for ai in range(self.num_anchors):
                    offset = 1 + ai * 3
                    dx = float(out[offset, cy, cx]) if offset < out.shape[0] else 0.0
                    dy = float(out[offset + 1, cy, cx]) if offset + 1 < out.shape[0] else 0.0
                    log_delta = float(out[offset + 2, cy, cx]) if offset + 2 < out.shape[0] else 0.0

                    anchor_size = self.anchor_scales[ai] * self.stride
                    center_x = (cx + 0.5 + dx) * self.stride / scale
                    center_y = (cy + 0.5 + dy) * self.stride / scale
                    box_size = anchor_size * np.exp(log_delta) / scale
                    box_size = max(box_size, 10.0)

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

        all_detections = []
        for sd in scale_detections:
            all_detections.extend(sd)

        if not all_detections:
            return []

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
                    is_different_face = True
                    for k in kept:
                        if compute_iou(d.bbox, k.bbox) >= 0.1:
                            is_different_face = False
                            break
                    if is_different_face:
                        kept.append(d)
            return kept
        return detections


# ============================================================================
# FaceFCN v5.0 — Full-frame anchor-free FPN architecture
# ============================================================================

class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                   padding=dilation, dilation=dilation,
                                   groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class FPNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DSConvBlock(32, 64, stride=1),
            DSConvBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            DSConvBlock(64, 128, stride=2),
            DSConvBlock(128, 128, stride=1),
            DSConvBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            DSConvBlock(128, 256, stride=2),
            DSConvBlock(256, 256, stride=1),
            DSConvBlock(256, 256, stride=1),
        )
        self.stage4 = nn.Sequential(
            DSConvBlock(256, 256, stride=1, dilation=2),
            DSConvBlock(256, 256, stride=1, dilation=2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)          # 640→320, stride 2
        c2 = self.stage1(x)       # 320→320, stride 2
        c3 = self.stage2(c2)      # 320→160, stride 4
        c4 = self.stage3(c3)      # 160→80,  stride 8
        c5 = self.stage4(c4)      # 80→80,   stride 8
        return c2, c3, c4, c5


class FPN(nn.Module):
    def __init__(self, fpn_dim: int = 64):
        super().__init__()
        self.lat2 = nn.Conv2d(64, fpn_dim, 1)    # C2(64ch, s=2) → P2
        self.lat3 = nn.Conv2d(128, fpn_dim, 1)   # C3(128ch, s=4) → P3
        self.lat4 = nn.Conv2d(256, fpn_dim, 1)   # C4(256ch, s=8) → P4
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c2, c3, c4):
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + nn.functional.interpolate(
            p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + nn.functional.interpolate(
            p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return p2, p3, p4


class AnchorFreeHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.obj_pred = nn.Conv2d(in_ch, 1, 1)
        self.bbox_pred = nn.Conv2d(in_ch, 4, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.obj_pred.bias, -2.5)

    def forward(self, x):
        return self.obj_pred(x), self.bbox_pred(x)


class FaceFCNv5(nn.Module):
    """FaceFCN v5.0 — Full-frame anchor-free FPN architecture.
    
    Feature map strides (from 640×480 input):
      P2: stride 2 (320×240 grid)  — small faces (5-20px)
      P3: stride 4 (160×120 grid)  — medium faces (16-64px)
      P4: stride 8 (80×60 grid)    — large faces (32-128px)
    
    Each grid cell predicts:
      - objectness logit (1 channel): sigmoid-activated at inference
      - bbox offsets (4 channels): dx, dy, dw, dh
        dx/dy: offset from cell center, range ~[-1, 1]
        dw/dh: log of face size relative to stride
    
    Output: dict with keys p2_obj/p2_bbox, p3_obj/p3_bbox, p4_obj/p4_bbox
    Each tensor shape: (B, 1, H, W) for obj, (B, 4, H, W) for bbox
    """
    strides = {"p2": 2, "p3": 4, "p4": 8}

    def __init__(self):
        super().__init__()
        self.backbone = FPNBackbone()
        self.fpn = FPN()
        self.head_p2 = AnchorFreeHead(64)
        self.head_p3 = AnchorFreeHead(64)
        self.head_p4 = AnchorFreeHead(64)

    def forward(self, x):
        c2, c3, c4, _ = self.backbone(x)
        p2, p3, p4 = self.fpn(c2, c3, c4)
        o2, b2 = self.head_p2(p2)
        o3, b3 = self.head_p3(p3)
        o4, b4 = self.head_p4(p4)
        return {
            "p2_obj": o2, "p2_bbox": b2,
            "p3_obj": o3, "p3_bbox": b3,
            "p4_obj": o4, "p4_bbox": b4,
        }


class FaceCNNv5:
    """v5.0 inference wrapper with peak-finding post-processing.
    
    Key differences from v4.0 FaceCNN:
    - Single forward pass (no 5-scale pyramid): ~500 MFLOPs vs 6.41 GFLOPs
    - Anchor-free head: peak-finding replaces per-anchor decode (14,400→~50 candidates)
    - FPN neck: 3 output levels at strides 2, 4, 8
    - Post-processing: morphological peak detection + light NMS, <5ms vs 100-500ms
    """
    def __init__(self, model_path: str = "models/face_cnn_v5.pth",
                 confidence_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.3,
                 per_level_thresholds: dict = None):
        """Face detection inference wrapper with per-FPN-level threshold support.

        Args:
            model_path: Path to trained FaceCNN checkpoint.
            confidence_threshold: Default global threshold (used if
                per_level_thresholds is None).
            nms_iou_threshold: IoU threshold for greedy NMS.
            per_level_thresholds: Dict of {level: sigmoid_threshold} for
                per-FPN-level peak finding. E.g. {"p4": 0.10, "p3": 0.30, "p2": 0.55}.
                If None, confidence_threshold is used uniformly for all levels.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.per_level_thresholds = per_level_thresholds or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = FaceFCNv5()
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(state_dict, dict) and 'ema_state_dict' in state_dict:
                state_dict = state_dict['ema_state_dict']
            elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            base_model.load_state_dict(state_dict, strict=True)
            base_model.eval()
            try:
                self.model = torch.jit.script(base_model)
                print(f"FaceCNNv5 loaded from {model_path} (TorchScript optimized)")
            except Exception as js_err:
                self.model = base_model
                print(f"FaceCNNv5 loaded from {model_path} "
                      f"(TorchScript unavailable: {js_err})")
            self.model.to(self.device)
            self._sanity_check(self.model)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FaceCNNv5 model not found at {model_path}. "
                "Training is required before running detection."
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FaceCNNv5 model from {model_path}: {e}. "
                "Ensure the model file is valid and compatible."
            ) from e

    def _peak_finding(self, obj: np.ndarray, threshold: float = None) -> np.ndarray:
        """Find local maxima in heatmap via morphological dilation.

        A cell is a peak if its value equals the dilated value AND its sigmoid
        probability exceeds the confidence threshold.

        Args:
            obj: Raw logit heatmap (pre-sigmoid).
            threshold: Sigmoid confidence threshold [0, 1]. If None,
                uses self.confidence_threshold. Converted to logit
                internally for comparison.

        Cost: O(n) over the grid, typically <0.1ms.
        """
        thresh = threshold if threshold is not None else self.confidence_threshold
        logit_thresh = float(np.log(thresh / (1.0 - thresh)))
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(obj, kernel)
        peaks = (obj == dilated) & (obj > logit_thresh)
        return peaks

    def detect(self, frame: np.ndarray) -> List[Face]:
        """Run detection on a single frame.
        
        Pipeline:
        1. Preprocess: BGR→RGB, normalize to [0,1]
        2. Forward pass: FaceFCNv5 produces 3 FPN levels
        3. Per level: peak finding, bbox decode, filter by size
        4. Cross-level NMS: greedy IoU-based merging
        
        Returns list of Face dataclasses sorted by confidence descending.
        Empty list if no detections above threshold.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        # FPN levels: (name, stride, expected face size range)
        levels = [
            ("p2", 2, out["p2_obj"][0, 0].cpu().numpy(), out["p2_bbox"][0].cpu().numpy()),
            ("p3", 4, out["p3_obj"][0, 0].cpu().numpy(), out["p3_bbox"][0].cpu().numpy()),
            ("p4", 8, out["p4_obj"][0, 0].cpu().numpy(), out["p4_bbox"][0].cpu().numpy()),
        ]

        all_faces: List[Face] = []
        for lname, stride, obj_map, bbox_map in levels:
            thresh = self.per_level_thresholds.get(
                lname, self.confidence_threshold)
            peaks = self._peak_finding(obj_map, threshold=thresh)
            if not peaks.any():
                continue

            ys, xs = np.where(peaks)
            confs = 1.0 / (1.0 + np.exp(-obj_map[ys, xs]))

            for cy, cx, conf in zip(ys, xs, confs):
                dx = float(bbox_map[0, cy, cx])
                dy = float(bbox_map[1, cy, cx])
                dw = float(bbox_map[2, cy, cx])
                dh = float(bbox_map[3, cy, cx])

                # Decode bbox from anchor-free offsets
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

                all_faces.append(Face(
                    bbox=BoundingBox(x=x1, y=y1, w=bw, h=bh),
                    confidence=float(conf)
                ))

        if not all_faces:
            return []

        # Greedy NMS: keep highest-confidence, suppress overlapping
        all_faces.sort(key=lambda f: f.confidence, reverse=True)
        kept: List[Face] = []
        for face in all_faces:
            keep = True
            for k in kept:
                if compute_iou(face.bbox, k.bbox) > self.nms_iou_threshold:
                    keep = False
                    break
            if keep:
                kept.append(face)

        return kept

    def _sanity_check(self, model: nn.Module) -> None:
        import warnings

        collapse_threshold = 1e-5
        content_delta_threshold = 1e-5
        warning_threshold = 0.10

        model.eval()
        with torch.no_grad():
            zero_in = torch.zeros(1, 3, 480, 640, device=self.device)
            out_zero = model(zero_in)
            if 'p4_obj' in out_zero:
                obj_zero = torch.sigmoid(out_zero['p4_obj'])
            else:
                obj_keys = [k for k in out_zero if 'obj' in k.lower()]
                obj_zero = torch.cat([torch.sigmoid(out_zero[k]) for k in obj_keys], dim=1)

            if not torch.isfinite(obj_zero).all():
                nans = torch.isnan(obj_zero).sum().item()
                infs = torch.isinf(obj_zero).sum().item()
                raise RuntimeError(
                    f"Model health FAILED: Output contains NaN={nans}, Inf={infs}"
                )

            obj_std = obj_zero.std().item()
            if obj_std < collapse_threshold:
                unique_vals = obj_zero.unique().numel()
                raise RuntimeError(
                    f"Model health FAILED: Output collapsed "
                    f"(std={obj_std:.2e} < {collapse_threshold:.1e}, "
                    f"unique values={unique_vals}). "
                    f"Causes: dead head, frozen BN stats, or EMA BN buffer bug."
                )

            one_in = torch.ones(1, 3, 480, 640, device=self.device)
            out_one = model(one_in)
            if 'p4_obj' in out_one:
                obj_one = torch.sigmoid(out_one['p4_obj'])
            else:
                obj_one = torch.cat([torch.sigmoid(out_one[k]) for k in obj_keys], dim=1)

            max_abs_diff = (obj_one - obj_zero).abs().max().item()
            if max_abs_diff < content_delta_threshold:
                raise RuntimeError(
                    f"Model health FAILED: No content sensitivity "
                    f"(max|f(white) - f(black)| = {max_abs_diff:.2e}). "
                    f"Cause: dead backbone."
                )

            max_sig = max(obj_zero.max().item(), obj_one.max().item())
            if max_sig < warning_threshold:
                warnings.warn(
                    f"Model health WARNING: Max sigmoid output = {max_sig:.4f} "
                    f"(below {warning_threshold:.2f} warning threshold). "
                    f"Model may produce zero detections at confidence_threshold=0.30. "
                    f"Consider re-training head (see FACECNN_v5_DEAD_HEAD_INVESTIGATION.md).",
                    category=UserWarning,
                )

            print(f"Model health CHECK PASSED:")
            print(f"  NaN/Inf check:     {'PASS' if torch.isfinite(obj_zero).all() else 'FAIL'}")
            print(f"  Output diversity:  PASS (std={obj_std:.2e} > {collapse_threshold:.1e})")
            print(f"  Content sensitive: PASS (delta={max_abs_diff:.4f})")
            print(f"  Detection ceiling: {'PASS' if max_sig >= warning_threshold else 'WARN'} "
                  f"(max sigmoid={max_sig:.4f}, threshold_30 {'pass' if max_sig >= 0.30 else 'FAIL'})")


class FaceCNNv7_1:
    """V7.1 inference wrapper with peak-finding + Soft-NMS.
    Drop-in replacement for FaceCNN with identical detect() interface.
    """
    def __init__(self, model_path: str = "models/face_cnn_v71.pth",
                 confidence_threshold: float = 0.25,
                 nms_iou_threshold: float = 0.3,
                 per_level_thresholds: dict = None):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.per_level_thresholds = per_level_thresholds or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = FaceFCNv7_1Model()
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(state_dict, dict) and 'ema_state_dict' in state_dict:
                state_dict = state_dict['ema_state_dict']
            elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            base_model.load_state_dict(state_dict, strict=True)
            base_model.eval()
            base_model.sanity_check()
            try:
                self.model = torch.jit.script(base_model)
                print(f"FaceCNNv7_1 loaded from {model_path} (TorchScript optimized)")
            except Exception as js_err:
                self.model = base_model
                print(f"FaceCNNv7_1 loaded from {model_path} "
                      f"(TorchScript unavailable: {js_err})")
            self.model.to(self.device)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FaceCNNv7_1 model not found at {model_path}. "
                "Training is required before running detection."
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FaceCNNv7_1 model from {model_path}: {e}. "
                "Ensure the model file is valid and compatible."
            ) from e

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        stride_map = {"": 4, "p2_": 2}
        thresh_map = {"": self.confidence_threshold,
                      "p2_": self.per_level_thresholds.get("p2", self.confidence_threshold * 0.6)}
        all_faces = []

        for prefix in ["", "p2_"]:
            obj_map = out[f"{prefix}obj"][0, 0].cpu().numpy()
            iou_map = out[f"{prefix}iou"][0, 0].cpu().numpy()
            bbox_map = out[f"{prefix}bbox"][0].cpu().numpy()
            stride = stride_map[prefix]
            thresh = thresh_map[prefix]

            quality = np.sqrt(
                (1.0 / (1.0 + np.exp(-obj_map)))
                * (1.0 / (1.0 + np.exp(-iou_map)))
                + 1e-8
            )

            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(quality, kernel)
            peaks = (quality == dilated) & (quality > thresh)

        all_faces = []
        if peaks.any():
            ys, xs = np.where(peaks)
            for cy, cx in zip(ys, xs):
                q = float(quality[cy, cx])
                dx = float(bbox_map[0, cy, cx])
                dy = float(bbox_map[1, cy, cx])
                dw = float(bbox_map[2, cy, cx])
                dh = float(bbox_map[3, cy, cx])

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
                all_faces.append(Face(
                    bbox=BoundingBox(x=x1, y=y1, w=bw, h=bh),
                    confidence=q
                ))

        if not all_faces:
            return []

        all_faces.sort(key=lambda f: f.confidence, reverse=True)
        kept = []
        for face in all_faces:
            max_decay = 0.0
            for k in kept:
                iou = compute_iou(face.bbox, k.bbox)
                if iou > self.nms_iou_threshold:
                    c1, c2 = face.bbox, k.bbox
                    center_dist = (c1.center_x - c2.center_x)**2 + (c1.center_y - c2.center_y)**2
                    diag = max(c1.w, c2.w)**2 + max(c1.h, c2.h)**2
                    diou = iou - center_dist / (diag + 1e-8)
                    if diou > 0:
                        max_decay = max(max_decay, diou)
            if max_decay > 0:
                decayed = face.confidence * (1.0 - max_decay)
                if decayed > 0.01:
                    kept.append(Face(
                        bbox=BoundingBox(x=face.bbox.x, y=face.bbox.y,
                                         w=face.bbox.w, h=face.bbox.h),
                        confidence=decayed
                    ))
            else:
                kept.append(face)

        return kept
