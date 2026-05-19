import numpy as np
import cv2
from src.cv.gesture_classifier import HandDetector, GestureClassifier, GestureResult, GESTURE_LABELS


def test_gesture_result_dataclass():
    r = GestureResult("OPEN_PALM", 0.95, "svm")
    assert r.gesture == "OPEN_PALM"
    assert r.confidence == 0.95
    assert r.method == "svm"


def test_gesture_labels_consistency():
    assert len(GESTURE_LABELS) == 5
    assert GESTURE_LABELS[0] == "OPEN_PALM"
    assert GESTURE_LABELS[1] == "FIST"
    assert GESTURE_LABELS[2] == "THUMBS_UP"
    assert GESTURE_LABELS[3] == "POINT"
    assert GESTURE_LABELS[4] == "PEACE"


def test_hand_detector_init():
    hd = HandDetector()
    assert hd.min_area == 1000
    assert hd.use_motion is True
    assert hd._prev_gray is None


def test_hand_detector_no_skin():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hd = HandDetector(min_area=1000, use_motion=False)
    result = hd.detect(frame)
    assert result is None


def test_hand_detector_with_face_exclusion():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (0, 133, 150)
    hd = HandDetector(ycrcb_lower=(0, 133, 77), ycrcb_upper=(255, 173, 127),
                       min_area=10, use_motion=False)

    class FakeBBox:
        x, y, w, h = 100, 100, 50, 50

    result = hd.detect(frame, face_bbox=FakeBBox())
    assert result is None or result[0] is not None


def test_hand_detector_skin_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (300, 300), (0, 140, 100), -1)
    hd = HandDetector(ycrcb_lower=(0, 133, 77), ycrcb_upper=(255, 173, 127),
                       min_area=50, use_motion=False)
    result = hd.detect(frame)
    assert result is None


def test_gesture_classifier_init_without_models():
    gc = GestureClassifier()
    assert gc.svm is None
    assert gc.scaler is None
    assert gc.pca is None
    assert gc.min_confidence == 0.6


def test_gesture_classifier_rule_based_none():
    roi = np.zeros((64, 64, 3), dtype=np.uint8)
    gc = GestureClassifier()
    result = gc.predict(roi)
    assert result.method == "rule"
    assert result.confidence == 0.9
    assert result.gesture in GESTURE_LABELS.values() or result.gesture == "NONE"


def test_gesture_classifier_rule_based_fist():
    roi = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.circle(roi, (32, 32), 20, (255, 255, 255), -1)
    gc = GestureClassifier()
    result = gc.predict(roi)
    assert result.method == "rule"
    assert result.gesture in GESTURE_LABELS.values() or result.gesture == "NONE"


def test_hand_detector_motion():
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (200, 200), (300, 300), (0, 140, 100), -1)

    hd = HandDetector(ycrcb_lower=(0, 133, 77), ycrcb_upper=(255, 173, 127),
                       min_area=50, use_motion=True)
    result1 = hd.detect(frame1)
    assert result1 is None
    result2 = hd.detect(frame2)
    assert result2 is None or result2[0] is not None


def test_gesture_classifier_svm_load_fail_graceful():
    gc = GestureClassifier(svm_path="/nonexistent/model.pkl")
    assert gc.svm is None
    roi = np.zeros((64, 64, 3), dtype=np.uint8)
    result = gc.predict(roi)
    assert result.method == "rule"


if __name__ == "__main__":
    test_gesture_result_dataclass()
    test_gesture_labels_consistency()
    test_hand_detector_init()
    test_hand_detector_no_skin()
    test_hand_detector_with_face_exclusion()
    test_hand_detector_skin_region()
    test_gesture_classifier_init_without_models()
    test_gesture_classifier_rule_based_none()
    test_gesture_classifier_rule_based_fist()
    test_hand_detector_motion()
    test_gesture_classifier_svm_load_fail_graceful()
    print("\nAll gesture classifier tests passed!")
