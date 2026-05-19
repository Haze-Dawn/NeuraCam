import numpy as np
import time
from src.capture.camera import Frame, Camera


def test_frame_dataclass():
    data = np.zeros((480, 640, 3), dtype=np.uint8)
    ts = time.time()
    frame = Frame(data=data, timestamp=ts, frame_id=42)
    assert frame.frame_id == 42
    assert frame.timestamp == ts
    assert frame.data.shape == (480, 640, 3)
    np.testing.assert_array_equal(frame.data, data)


def test_frame_id_monotonic():
    f1 = Frame(np.zeros((10, 10, 3)), 0.0, 1)
    f2 = Frame(np.zeros((10, 10, 3)), 0.0, 2)
    assert f2.frame_id > f1.frame_id


def test_camera_init_no_source():
    try:
        cam = Camera(source=9999, use_capture_thread=False)
        cam.release()
    except Exception:
        pass


def test_camera_get_processing_frame():
    cam = Camera(source=9999, use_capture_thread=False)
    full_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    proc = cam.get_processing_frame(full_frame)
    assert proc.shape == (cam.processing_height, cam.processing_width, 3)
    cam.release()


def test_camera_processing_resize():
    cam = Camera(source=9999, use_capture_thread=False)
    full = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    proc = cam.get_processing_frame(full)
    expected_h = cam.processing_height
    expected_w = cam.processing_width
    assert proc.shape[0] == expected_h
    assert proc.shape[1] == expected_w
    assert proc.dtype == np.uint8
    cam.release()


def test_camera_read_no_source():
    cam = Camera(source=9999, use_capture_thread=False)
    result = cam.read()
    assert result is None
    cam.release()


def test_camera_fps_property():
    cam = Camera(source=9999, use_capture_thread=False)
    assert cam.fps == 30
    cam.release()


def test_camera_source_change():
    cam = Camera(source=0, use_capture_thread=False)
    assert cam.source == 0
    cam.release()


def test_camera_release_safe():
    cam = Camera(source=9999, use_capture_thread=False)
    cam.release()
    cam.release()


if __name__ == "__main__":
    test_frame_dataclass()
    test_frame_id_monotonic()
    test_camera_init_no_source()
    test_camera_get_processing_frame()
    test_camera_processing_resize()
    test_camera_read_no_source()
    test_camera_fps_property()
    test_camera_source_change()
    test_camera_release_safe()
    print("\nAll camera tests passed!")
