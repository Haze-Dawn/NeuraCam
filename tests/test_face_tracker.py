import numpy as np
from src.cv.face_tracker import KalmanTracker, BoundingBox, Face, compute_iou


def test_bounding_box():
    bbox = BoundingBox(x=10, y=20, w=100, h=150)
    assert bbox.center_x == 60.0
    assert bbox.center_y == 95.0
    print("PASS: BoundingBox center computation")


def test_face_creation():
    bbox = BoundingBox(10, 20, 100, 150)
    face = Face(bbox=bbox, confidence=0.95)
    assert face.confidence == 0.95
    assert face.bbox.w == 100
    print("PASS: Face creation")


def test_kalman_init():
    tracker = KalmanTracker(max_lost_frames=5, iou_threshold=0.3)
    assert tracker.max_lost_frames == 5
    assert tracker.iou_threshold == 0.3
    assert len(tracker.tracks) == 0
    print("PASS: KalmanTracker initialization")


def test_kalman_empty():
    tracker = KalmanTracker()
    result = tracker.update([])
    assert result is None
    print("PASS: KalmanTracker handles empty detections")


def test_kalman_single_detection():
    tracker = KalmanTracker()
    face = Face(BoundingBox(0, 0, 100, 100), 0.9)
    result = tracker.update([face])
    assert result is not None
    print("PASS: KalmanTracker accepts single detection")


def test_kalman_overlap():
    tracker = KalmanTracker(iou_threshold=0.3)
    face1 = Face(BoundingBox(0, 0, 100, 100), 0.9)
    face2 = Face(BoundingBox(90, 0, 100, 100), 0.9)
    result = tracker.update([face1])
    assert result is not None
    result2 = tracker.update([face2])
    assert result2 is not None
    print("PASS: KalmanTracker IoU matching")


def test_iou_zero():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(100, 100, 10, 10)
    iou = compute_iou(a, b)
    assert iou == 0.0
    print("PASS: IoU zero for non-overlapping boxes")


def test_iou_perfect():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(0, 0, 100, 100)
    iou = compute_iou(a, b)
    assert iou == 1.0
    print("PASS: IoU perfect overlap")


def test_kalman_track_persists():
    tracker = KalmanTracker(max_lost_frames=5)
    face = Face(BoundingBox(0, 0, 100, 100), 0.9)
    tracker.update([face])
    for _ in range(3):
        result = tracker.update([])
        assert result is not None
    print("PASS: KalmanTrack persists through brief occlusion")


def test_kalman_track_expires():
    tracker = KalmanTracker(max_lost_frames=2)
    face = Face(BoundingBox(0, 0, 100, 100), 0.9)
    tracker.update([face])
    for _ in range(3):
        tracker.update([])
    result = tracker.update([])
    assert result is None
    print("PASS: KalmanTrack expires after max_lost_frames")


if __name__ == "__main__":
    test_bounding_box()
    test_face_creation()
    test_kalman_init()
    test_kalman_empty()
    test_kalman_single_detection()
    test_kalman_overlap()
    test_iou_zero()
    test_iou_perfect()
    test_kalman_track_persists()
    test_kalman_track_expires()
    print("\nAll Kalman tracker tests passed!")
