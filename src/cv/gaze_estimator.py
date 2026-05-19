import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GazeResult:
    direction: int
    confidence: float
    method: str

    def __post_init__(self):
        labels = {0: "CENTER", 1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}
        self.direction_label = labels.get(self.direction, "UNKNOWN")


DIRECTION_LABELS = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]


class GazeEstimator:
    def __init__(self, custom_cnn_path: Optional[str] = None):
        self.custom_cnn = None
        self._face_mesh = None

        if custom_cnn_path:
            self.custom_cnn = self._load_custom_cnn(custom_cnn_path)

        if self.custom_cnn is None:
            self._init_geometric()

    def _load_custom_cnn(self, path: str):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.jit.load(path, map_location=device)
            model.eval()
            print(f"Loaded gaze CNN from {path}")
            return model
        except Exception as e:
            print(f"Failed to load gaze CNN: {e}")
            return None

    def _init_geometric(self):
        import mediapipe as mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def predict(self, face_crop: np.ndarray) -> GazeResult:
        if self.custom_cnn is not None:
            return self._predict_custom_cnn(face_crop)
        return self._predict_geometric(face_crop)

    def _predict_custom_cnn(self, crop: np.ndarray) -> GazeResult:
        try:
            import torch
            h, w = crop.shape[:2]
            if h < 10 or w < 10:
                return GazeResult(0, 0.0, "custom_cnn")
            img = cv2.resize(crop, (128, 128))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0)
            with torch.no_grad():
                logits = self.custom_cnn(tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = float(probs.max().item())
            return GazeResult(pred, conf, "custom_cnn")
        except Exception as e:
            print(f"Custom CNN inference failed: {e}")
            return GazeResult(0, 0.0, "custom_cnn")

    def _predict_geometric(self, crop: np.ndarray) -> GazeResult:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return GazeResult(0, 0.0, "geometric")
            landmarks = results.multi_face_landmarks[0]
            h, w = crop.shape[:2]

            left_eye_pts = [
                (int(landmarks.landmark[i].x * w),
                 int(landmarks.landmark[i].y * h))
                for i in [33, 133]
            ]
            right_eye_pts = [
                (int(landmarks.landmark[i].x * w),
                 int(landmarks.landmark[i].y * h))
                for i in [362, 263]
            ]
            left_iris = (
                int(landmarks.landmark[468].x * w),
                int(landmarks.landmark[468].y * h)
            )
            right_iris = (
                int(landmarks.landmark[473].x * w),
                int(landmarks.landmark[473].y * h)
            )

            l_eye_center = ((left_eye_pts[0][0] + left_eye_pts[1][0]) / 2,
                            (left_eye_pts[0][1] + left_eye_pts[1][1]) / 2)
            r_eye_center = ((right_eye_pts[0][0] + right_eye_pts[1][0]) / 2,
                            (right_eye_pts[0][1] + right_eye_pts[1][1]) / 2)
            l_eye_w = abs(left_eye_pts[1][0] - left_eye_pts[0][0])
            r_eye_w = abs(right_eye_pts[1][0] - right_eye_pts[0][0])

            gaze_x = ((left_iris[0] - l_eye_center[0]) / max(l_eye_w, 1) +
                      (right_iris[0] - r_eye_center[0]) / max(r_eye_w, 1)) / 2
            gaze_y = ((left_iris[1] - l_eye_center[1]) / max(l_eye_w, 1) +
                      (right_iris[1] - r_eye_center[1]) / max(r_eye_w, 1)) / 2

            return self._offset_to_class(gaze_x, gaze_y)
        except Exception:
            return GazeResult(0, 0.0, "geometric")

    def _offset_to_class(self, gaze_x: float, gaze_y: float) -> GazeResult:
        threshold = 0.15
        if abs(gaze_x) < threshold and abs(gaze_y) < threshold:
            conf = 1.0 - min(abs(gaze_x), abs(gaze_y))
            return GazeResult(0, conf, "geometric")
        if gaze_x < -threshold:
            return GazeResult(1, min(abs(gaze_x), 1.0), "geometric")
        if gaze_x > threshold:
            return GazeResult(2, min(abs(gaze_x), 1.0), "geometric")
        if gaze_y < -threshold:
            return GazeResult(3, min(abs(gaze_y), 1.0), "geometric")
        if gaze_y > threshold:
            return GazeResult(4, min(abs(gaze_y), 1.0), "geometric")
        return GazeResult(0, 0.3, "geometric")

    def __del__(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
