import cv2
import numpy as np
from typing import Optional
from src.cv.face_detector import Face, BoundingBox
from src.cv.gaze_estimator import GazeResult, DIRECTION_LABELS
from src.cv.gesture_classifier import GestureResult
from src.control.state_machine import Mode


def crop_face_region(frame: np.ndarray, bbox: BoundingBox,
                     margin: float = 0.3) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox.x, bbox.y, bbox.w, bbox.h
    mw, mh = int(bw * margin), int(bh * margin)
    x1 = max(0, x - mw)
    y1 = max(0, y - mh)
    x2 = min(w, x + bw + mw)
    y2 = min(h, y + bh + mh)
    return frame[y1:y2, x1:x2]


def compute_framing_error(face_bbox: BoundingBox, frame_size: tuple,
                          dead_zone: float = 0.05) -> tuple[float, float]:
    fw, fh = frame_size
    face_cx = face_bbox.x + face_bbox.w / 2
    face_cy = face_bbox.y + face_bbox.h / 2
    error_x = 2 * (face_cx / fw - 0.5)
    error_y = 2 * (face_cy / fh - 0.5)
    if abs(error_x) < dead_zone:
        error_x = 0
    if abs(error_y) < dead_zone:
        error_y = 0
    return error_x, error_y


def draw_debug_overlay(
    frame: np.ndarray,
    face: Optional[Face] = None,
    gaze: Optional[GazeResult] = None,
    gesture: Optional[GestureResult] = None,
    mode: Mode = Mode.IDLE,
    fps: float = 0.0,
    recording: bool = False,
    gimbal_angles: tuple = (90, 90),
    imu_angles: Optional[tuple] = None,
) -> np.ndarray:
    overlay = frame.copy()

    if face:
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(overlay, (int(face.bbox.center_x), int(face.bbox.center_y)),
                   4, (0, 255, 0), -1)

    if gaze:
        arrow_colors = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 0, 255),
            3: (0, 255, 255),
            4: (255, 255, 0),
        }
        color = arrow_colors.get(gaze.direction, (255, 255, 255))
        h, w = overlay.shape[:2]
        cx, cy = w // 2, h // 2
        arrow_end = (cx, cy)
        if gaze.direction == 0:
            cv2.circle(overlay, (cx, cy), 10, color, 2)
        elif gaze.direction == 1:
            arrow_end = (cx - 60, cy)
        elif gaze.direction == 2:
            arrow_end = (cx + 60, cy)
        elif gaze.direction == 3:
            arrow_end = (cx, cy - 60)
        elif gaze.direction == 4:
            arrow_end = (cx, cy + 60)
        cv2.arrowedLine(overlay, (cx, cy), arrow_end, color, 2, tipLength=0.3)
        cv2.putText(overlay, f"Gaze: {DIRECTION_LABELS[gaze.direction]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if gesture and gesture.gesture != "NONE":
        color = (0, 255, 255) if gesture.method == "rule" else (255, 0, 255)
        cv2.putText(overlay, f"Gesture: {gesture.gesture}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    mode_colors = {
        Mode.IDLE: (128, 128, 128),
        Mode.TRACKING: (0, 255, 0),
        Mode.LOCKED: (0, 165, 255),
        Mode.HOME: (255, 255, 0),
    }
    mode_color = mode_colors.get(mode, (255, 255, 255))
    cv2.rectangle(overlay, (5, 5), (200, 50), (0, 0, 0), -1)
    cv2.putText(overlay, f"[{mode.value}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    if fps > 0:
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if recording:
        cv2.putText(overlay, "REC", (overlay.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    pan, tilt = gimbal_angles
    cv2.putText(overlay, f"Pan:{pan} Tilt:{tilt}",
                (10, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if imu_angles is not None:
        pitch, roll, yaw = imu_angles
        cv2.putText(overlay, f"IMU P:{pitch:.1f} R:{roll:.1f} Y:{yaw:.1f}",
                    (10, overlay.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return overlay
