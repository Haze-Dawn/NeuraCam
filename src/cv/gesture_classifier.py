import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GestureResult:
    gesture: str
    confidence: float
    method: str


GESTURE_LABELS = ["OPEN_PALM", "FIST", "THUMBS_UP", "POINT", "PEACE"]
GESTURE_ACTIONS = {"OPEN_PALM", "FIST", "THUMBS_UP"}
WINDOW_SIZE = (64, 64)


class GestureClassifier:
    def __init__(self, svm_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 min_confidence: float = 0.6):
        self.svm = None
        self.scaler = None
        self.min_confidence = min_confidence

        self.hog = cv2.HOGDescriptor(
            _winSize=WINDOW_SIZE,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

        if svm_path:
            try:
                import joblib
                self.svm = joblib.load(svm_path)
                self.scaler = joblib.load(scaler_path) if scaler_path else None
                print(f"Loaded gesture SVM from {svm_path}")
            except Exception as e:
                print(f"Failed to load gesture SVM: {e}")

    def find_hand_roi(self, frame: np.ndarray,
                      face_bbox: Optional[tuple] = None) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 40])
        upper = np.array([25, 170, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            margin = int(max(fw, fh) * 0.3)
            x1 = max(0, fx - margin)
            y1 = max(0, fy - margin)
            x2 = min(frame.shape[1], fx + fw + margin)
            y2 = min(frame.shape[0], fy + fh + margin)
            mask[y1:y2, x1:x2] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) < 2000:
            return None

        x, y, w, h = cv2.boundingRect(hand)
        margin = int(max(w, h) * 0.1)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2]

    def extract_hog_features(self, hand_roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, WINDOW_SIZE)
        features = self.hog.compute(resized).ravel()
        return features

    def predict(self, frame: np.ndarray,
                face_bbox: Optional[tuple] = None) -> GestureResult:
        hand_roi = self.find_hand_roi(frame, face_bbox)
        if hand_roi is None:
            return GestureResult("NONE", 0.0, "none")

        features = self.extract_hog_features(hand_roi)

        if self.svm is not None:
            try:
                X = features.reshape(1, -1)
                if self.scaler:
                    X = self.scaler.transform(X)
                pred = self.svm.predict(X)[0]
                conf = float(np.max(self.svm.predict_proba(X)))
                gesture = GESTURE_LABELS[pred]
                if conf >= self.min_confidence:
                    return GestureResult(gesture, conf, "svm")
            except Exception as e:
                print(f"SVM prediction failed: {e}")

        return self._rule_based(hand_roi)

    def _rule_based(self, hand_roi: np.ndarray) -> GestureResult:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GestureResult("NONE", 0.0, "rule")

        hand = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(hand, returnPoints=False)
        if hull.ndim == 1 or len(hull) < 4:
            return GestureResult("FIST", 0.7, "rule")

        defects = cv2.convexityDefects(hand, hull)
        if defects is None:
            return GestureResult("FIST", 0.7, "rule")

        ext_fingers = 0
        for i in range(defects.shape[0]):
            _, _, far_dist = defects[i, 0, 2], defects[i, 0, 3]
            far_dist = far_dist / 256.0
            if far_dist > 15:
                ext_fingers += 1

        if ext_fingers >= 4:
            return GestureResult("OPEN_PALM", 0.8, "rule")
        elif ext_fingers <= 1:
            return GestureResult("FIST", 0.8, "rule")
        elif ext_fingers == 2:
            return GestureResult("PEACE", 0.7, "rule")
        elif ext_fingers == 3:
            return GestureResult("POINT", 0.7, "rule")
        return GestureResult("NONE", 0.5, "rule")

    def close(self):
        pass

    def __del__(self):
        pass
