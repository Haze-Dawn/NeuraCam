import cv2
import numpy as np
from typing import Optional, Tuple
from src.cv.face_tracker import Face, BoundingBox
from src.cv.gesture_classifier import GestureResult
from src.control.state_machine import Mode


def compute_framing_error(face_bbox: BoundingBox, frame_size: Tuple[int, int],
                          dead_zone: float = 0.05) -> Tuple[float, float]:
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
    gesture: Optional[GestureResult] = None,
    mode: Mode = Mode.IDLE,
    fps: float = 0.0,
    recording: bool = False,
    gimbal_angles: Tuple[int, int] = (90, 90),
    imu_angles: Optional[Tuple[float, float, float]] = None,
    kalman_uncertainty: float = 0.0,
    zoom_level: float = 1.0,
    tracking_target: str = "FACE",
) -> np.ndarray:
    overlay = frame.copy()

    if face:
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(overlay, (int(face.bbox.center_x), int(face.bbox.center_y)),
                   4, (0, 255, 0), -1)

    if gesture and gesture.gesture != "NONE":
        color = (0, 255, 255) if gesture.method == "rule" else (255, 0, 255)
        cv2.putText(overlay, f"Gesture: {gesture.gesture}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    mode_colors = {
        Mode.IDLE: (128, 128, 128),
        Mode.TRACKING: (0, 255, 0),
        Mode.TRACKING_HAND: (255, 165, 0),
        Mode.LOCKED: (0, 165, 255),
        Mode.HOME: (255, 255, 0),
        Mode.SEARCH: (255, 128, 0),
    }
    mode_color = mode_colors.get(mode, (255, 255, 255))
    target_suffix = f" ({tracking_target})" if mode in (
        Mode.TRACKING, Mode.TRACKING_HAND, Mode.LOCKED) else ""
    cv2.rectangle(overlay, (5, 5), (250, 50), (0, 0, 0), -1)
    cv2.putText(overlay, f"[{mode.value}{target_suffix}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    if fps > 0:
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if zoom_level > 1.0:
        cv2.putText(overlay, f"ZOOM: {zoom_level:.1f}x", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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

    if kalman_uncertainty > 0:
        bar_x = overlay.shape[1] - 120
        bar_y = 60
        bar_w = 100
        bar_h = 8
        fill_w = int(bar_w * min(kalman_uncertainty, 1.0))
        color = (0, 255, 0) if kalman_uncertainty < 0.3 else \
                (0, 255, 255) if kalman_uncertainty < 0.6 else (0, 0, 255)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), 1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      color, -1)

    return overlay
