import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2


@dataclass
class Face:
    bbox: BoundingBox
    landmarks: list
    confidence: float


class FaceDetector:
    def __init__(self, model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 min_confidence: float = 0.0,
                 window_size: tuple = (64, 64),
                 scale_factor: float = 1.15,
                 step_size: int = 4):
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.step_size = step_size
        self.svm = None
        self.scaler = None

        self.hog = cv2.HOGDescriptor(
            _winSize=window_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

        if model_path:
            self.load(model_path, scaler_path)

    def load(self, model_path: str, scaler_path: Optional[str] = None):
        import joblib
        try:
            self.svm = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path) if scaler_path else None
            print(f"Loaded face SVM from {model_path}")
        except Exception as e:
            print(f"Failed to load face SVM: {e}")

    def detect(self, frame: np.ndarray) -> list[Face]:
        if self.svm is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        win_h, win_w = self.window_size

        detections = []

        scale = 1.0
        while min(h, w) / scale >= min(win_h, win_w):
            scaled_h = int(h / scale)
            scaled_w = int(w / scale)
            scaled = cv2.resize(gray, (scaled_w, scaled_h))

            for y in range(0, scaled_h - win_h + 1, self.step_size):
                for x in range(0, scaled_w - win_w + 1, self.step_size):
                    patch = scaled[y:y + win_h, x:x + win_w]
                    features = self.hog.compute(patch).reshape(1, -1)

                    if self.scaler:
                        features = self.scaler.transform(features)

                    pred = self.svm.predict(features)[0]
                    conf = float(self.svm.decision_function(features)[0])

                    if pred == 1 and conf >= self.min_confidence:
                        orig_x = int(x * scale)
                        orig_y = int(y * scale)
                        orig_w = int(win_w * scale)
                        orig_h = int(win_h * scale)
                        detections.append({
                            'bbox': BoundingBox(orig_x, orig_y, orig_w, orig_h),
                            'confidence': conf,
                        })

            scale *= self.scale_factor

        detections = self._nms(detections, 0.3)

        return [
            Face(bbox=d['bbox'], landmarks=[], confidence=d['confidence'])
            for d in detections
        ]

    def _nms(self, detections: list, overlap_threshold: float = 0.3) -> list:
        if not detections:
            return []

        boxes = np.array([[d['bbox'].x, d['bbox'].y,
                           d['bbox'].x + d['bbox'].w,
                           d['bbox'].y + d['bbox'].h]
                          for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(iou <= overlap_threshold)[0] + 1]

        return [detections[i] for i in keep]


class FaceTracker:
    def __init__(self, max_lost_frames: int = 5, iou_threshold: float = 0.3):
        self.max_lost = max_lost_frames
        self.iou_threshold = iou_threshold
        self._tracks = {}
        self._next_id = 0
        self._lost_counts = {}
        self._histories = {}
        self._primary_id: Optional[int] = None

    def update(self, detections: list[Face]) -> Optional[Face]:
        if not detections:
            for tid in list(self._tracks.keys()):
                self._lost_counts[tid] = self._lost_counts.get(tid, 0) + 1
                if self._lost_counts[tid] > self.max_lost:
                    del self._tracks[tid]
                    self._lost_counts.pop(tid, None)
                    self._histories.pop(tid, None)
                    if self._primary_id == tid:
                        self._primary_id = None
            if self._primary_id is not None and self._primary_id in self._tracks:
                return self._tracks[self._primary_id]
            return None

        matched = set()
        det_used = [False] * len(detections)

        for tid in sorted(self._tracks.keys()):
            best_iou = self.iou_threshold
            best_idx = -1
            for i, det in enumerate(detections):
                if det_used[i]:
                    continue
                iou = self._compute_iou(
                    self._tracks[tid].bbox, det.bbox
                )
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                self._tracks[tid] = detections[best_idx]
                self._lost_counts[tid] = 0
                self._histories.setdefault(tid, []).append(detections[best_idx])
                if len(self._histories[tid]) > 30:
                    self._histories[tid] = self._histories[tid][-30:]
                det_used[best_idx] = True
                matched.add(tid)

        for i, det in enumerate(detections):
            if not det_used[i]:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = det
                self._lost_counts[tid] = 0
                self._histories[tid] = [det]

        unmatched = set(self._tracks.keys()) - matched
        for tid in unmatched:
            self._lost_counts[tid] = self._lost_counts.get(tid, 0) + 1
            if self._lost_counts[tid] > self.max_lost:
                del self._tracks[tid]
                self._lost_counts.pop(tid, None)
                self._histories.pop(tid, None)

        self._select_primary()
        if self._primary_id is not None and self._primary_id in self._tracks:
            return self._tracks[self._primary_id]
        return None

    def _select_primary(self):
        if not self._tracks:
            self._primary_id = None
            return
        self._primary_id = max(
            self._tracks.keys(),
            key=lambda tid: len(self._histories.get(tid, []))
        )

    def _compute_iou(self, a: BoundingBox, b: BoundingBox) -> float:
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.w, b.y + b.h

        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = a.w * a.h
        area_b = b.w * b.h
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
