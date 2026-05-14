import numpy as np
from src.cv.face_detector import FaceDetector, FaceTracker, BoundingBox, Face


def test_bounding_box():
    bbox = BoundingBox(x=10, y=20, w=100, h=150)
    assert bbox.center_x == 60.0
    assert bbox.center_y == 95.0
    print("PASS: BoundingBox center computation")


def test_face_creation():
    bbox = BoundingBox(10, 20, 100, 150)
    face = Face(bbox=bbox, landmarks=[], confidence=0.95)
    assert face.confidence == 0.95
    assert face.bbox.w == 100
    print("PASS: Face creation")


def test_face_detector_init():
    detector = FaceDetector(min_confidence=0.5)
    assert detector.min_confidence == 0.5
    print("PASS: FaceDetector initialization")


def test_face_tracker_init():
    tracker = FaceTracker(max_lost_frames=5, iou_threshold=0.3)
    assert tracker.max_lost == 5
    assert tracker.iou_threshold == 0.3
    print("PASS: FaceTracker initialization")


def test_face_tracker_empty():
    tracker = FaceTracker()
    result = tracker.update([])
    assert result is None
    print("PASS: FaceTracker handles empty detections")


def test_face_tracker_iou():
    tracker = FaceTracker(iou_threshold=0.3)
    face1 = Face(BoundingBox(0, 0, 100, 100), [], 0.9)
    face2 = Face(BoundingBox(90, 0, 100, 100), [], 0.9)
    result = tracker.update([face1])
    assert result is not None
    result2 = tracker.update([face2])
    assert result2 is not None
    print("PASS: FaceTracker IoU matching")


def test_iou_zero():
    tracker = FaceTracker()
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(100, 100, 10, 10)
    iou = tracker._compute_iou(a, b)
    assert iou == 0.0
    print("PASS: IoU zero for non-overlapping boxes")


if __name__ == "__main__":
    test_bounding_box()
    test_face_creation()
    test_face_detector_init()
    test_face_tracker_init()
    test_face_tracker_empty()
    test_face_tracker_iou()
    test_iou_zero()
    print("\nAll face detector tests passed!")
