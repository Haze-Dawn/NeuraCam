import numpy as np
from src.utils.visualization import compute_framing_error, draw_debug_overlay
from src.cv.face_tracker import Face, BoundingBox
from src.cv.gesture_classifier import GestureResult
from src.control.state_machine import Mode


def test_compute_framing_error_centered():
    bbox = BoundingBox(x=540, y=230, w=200, h=200)
    ex, ey = compute_framing_error(bbox, (1280, 720), dead_zone=0.05)
    cx = bbox.x + bbox.w / 2
    cy = bbox.y + bbox.h / 2
    expected_x = 2 * (cx / 1280 - 0.5)
    expected_y = 2 * (cy / 720 - 0.5)
    assert abs(ex - expected_x) < 0.001 or ex == 0
    assert abs(ey - expected_y) < 0.001 or ey == 0


def test_compute_framing_error_dead_zone():
    bbox = BoundingBox(x=630, y=350, w=20, h=20)
    ex, ey = compute_framing_error(bbox, (1280, 720), dead_zone=0.1)
    assert ex == 0
    assert ey == 0


def test_compute_framing_error_no_dead_zone():
    bbox = BoundingBox(x=0, y=0, w=100, h=100)
    ex, ey = compute_framing_error(bbox, (1280, 720), dead_zone=0.0)
    assert ex != 0 or ey != 0


def test_compute_framing_error_off_center():
    bbox = BoundingBox(x=0, y=0, w=50, h=50)
    ex, ey = compute_framing_error(bbox, (1280, 720), dead_zone=0.0)
    assert ex < 0
    assert ey < 0


def test_debug_overlay_no_face():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_debug_overlay(frame)
    assert overlay.shape == frame.shape
    assert overlay.dtype == np.uint8


def test_debug_overlay_with_face():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face = Face(BoundingBox(100, 100, 200, 200), 0.95)
    overlay = draw_debug_overlay(frame, face=face)
    assert overlay.shape == frame.shape


def test_debug_overlay_with_gesture():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    gesture = GestureResult("OPEN_PALM", 0.95, "svm")
    overlay = draw_debug_overlay(frame, gesture=gesture)
    assert overlay.shape == frame.shape


def test_debug_overlay_mode_colors():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for mode in [Mode.IDLE, Mode.TRACKING, Mode.LOCKED, Mode.HOME, Mode.SEARCH]:
        overlay = draw_debug_overlay(frame.copy(), mode=mode)
        assert overlay.shape == frame.shape


def test_debug_overlay_recording():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_debug_overlay(frame, recording=True)
    assert overlay.shape == frame.shape


def test_debug_overlay_gimbal_angles():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_debug_overlay(frame, gimbal_angles=(45, 90))
    assert overlay.shape == frame.shape


def test_debug_overlay_imu():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_debug_overlay(frame, imu_angles=(1.5, -0.5, 0.0))
    assert overlay.shape == frame.shape


def test_debug_overlay_kalman_uncertainty():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_debug_overlay(frame, kalman_uncertainty=0.5)
    assert overlay.shape == frame.shape


if __name__ == "__main__":
    test_compute_framing_error_centered()
    test_compute_framing_error_dead_zone()
    test_compute_framing_error_no_dead_zone()
    test_compute_framing_error_off_center()
    test_debug_overlay_no_face()
    test_debug_overlay_with_face()
    test_debug_overlay_with_gesture()
    test_debug_overlay_mode_colors()
    test_debug_overlay_recording()
    test_debug_overlay_gimbal_angles()
    test_debug_overlay_imu()
    test_debug_overlay_kalman_uncertainty()
    print("\nAll visualization tests passed!")
