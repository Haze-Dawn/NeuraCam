import cv2
import numpy as np
import joblib
from typing import Optional, Tuple
from dataclasses import dataclass

GESTURE_LABELS = {0: "OPEN_PALM", 1: "FIST", 2: "THUMBS_UP", 3: "POINT", 4: "PEACE"}
GESTURE_ACTIONS = {"OPEN_PALM", "FIST", "THUMBS_UP"}


@dataclass
class GestureResult:
    gesture: str
    confidence: float
    method: str


class HandDetector:
    def __init__(self,
                 ycrcb_lower: Tuple[int, int, int] = (0, 133, 77),
                 ycrcb_upper: Tuple[int, int, int] = (255, 173, 127),
                 hsv_lower: Tuple[int, int, int] = (0, 30, 60),
                 hsv_upper: Tuple[int, int, int] = (20, 150, 255),
                 min_area: int = 1000,
                 use_motion: bool = True):
        self.ycrcb_lower = np.array(ycrcb_lower)
        self.ycrcb_upper = np.array(ycrcb_upper)
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area
        self.use_motion = use_motion
        self._prev_gray: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray,
               face_bbox=None) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        if self.use_motion and self._prev_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, self._prev_gray)
            _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            skin_mask = cv2.bitwise_and(skin_mask, motion_mask)
            self._prev_gray = gray
        elif self.use_motion:
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_bbox is not None:
            x, y, w, h = face_bbox.x, face_bbox.y, face_bbox.w, face_bbox.h
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            fx1 = max(0, x - margin_x)
            fy1 = max(0, y - margin_y)
            fx2 = min(frame.shape[1], x + w + margin_x)
            fy2 = min(frame.shape[0], y + h + margin_y)
            face_exclusion = np.zeros(skin_mask.shape, dtype=np.uint8)
            cv2.rectangle(face_exclusion, (fx1, fy1), (fx2, fy2), 255, -1)
            skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(face_exclusion))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        return roi, (x, y, w, h)


class GestureClassifier:
    def __init__(self, svm_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 pca_path: Optional[str] = None,
                 min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.svm = None
        self.scaler = None
        self.pca = None
        self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        self.last_defect_count = 0

        if svm_path:
            try:
                self.svm = joblib.load(svm_path)
            except Exception as e:
                print(f"Failed to load SVM: {e}")
        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Failed to load scaler: {e}")
        if pca_path:
            try:
                self.pca = joblib.load(pca_path)
            except Exception as e:
                print(f"Failed to load PCA: {e}")

    def compute_defect_count(self, roi: np.ndarray) -> int:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        hand_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if hull.ndim == 1 or len(hull) < 3:
            return 0
        defects = cv2.convexityDefects(hand_contour, hull)
        if defects is None:
            return 0
        count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 15000:
                count += 1
        return count

    def predict(self, hand_roi: np.ndarray) -> GestureResult:
        roi = cv2.resize(hand_roi, (64, 64))
        features = self.hog.compute(roi).flatten()

        if self.svm is not None and self.scaler is not None:
            try:
                X = features.reshape(1, -1)
                if self.pca is not None:
                    X = self.pca.transform(X)
                X_scaled = self.scaler.transform(X)
                pred = self.svm.predict(X_scaled)[0]
                proba = self.svm.predict_proba(X_scaled)[0]
                confidence = float(max(proba))
                if confidence >= self.min_confidence:
                    gesture = GESTURE_LABELS.get(int(pred), "NONE")
                    return GestureResult(gesture, confidence, "svm")
            except Exception as e:
                print(f"SVM predict failed: {e}")

        gesture = self._rule_based(roi)
        return GestureResult(gesture, 0.9, "rule")

    def _rule_based(self, roi: np.ndarray) -> str:
        self.last_defect_count = self.compute_defect_count(roi)
        defect_count = self.last_defect_count

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hand_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(hand_contour)
            aspect_ratio = h / max(w, 1)
        else:
            aspect_ratio = 1.0

        if defect_count >= 4:
            return "OPEN_PALM"
        elif defect_count == 0:
            if aspect_ratio > 1.4:
                return "POINT"
            else:
                return "FIST"
        elif defect_count == 1:
            if aspect_ratio < 1.1:
                return "THUMBS_UP"
            elif aspect_ratio > 1.6:
                return "POINT"
            else:
                return "THUMBS_UP"
        elif defect_count == 2:
            return "PEACE"
        else:
            return "OPEN_PALM"
