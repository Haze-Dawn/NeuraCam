import numpy as np
import os
from src.capture.recorder import Recorder


def test_recorder_init():
    r = Recorder()
    assert r.recording is False
    assert r.fps == 30.0
    assert r.width == 1280
    assert r.height == 720


def test_recorder_start_stop():
    r = Recorder(output_path="/tmp/test_recording.mp4")
    assert r.recording is False
    r.start()
    assert r.recording is True
    r.stop()
    assert r.recording is False
    if os.path.exists("/tmp/test_recording.mp4"):
        os.remove("/tmp/test_recording.mp4")


def test_recorder_write_frame():
    r = Recorder(output_path="/tmp/test_recording_write.mp4")
    r.start()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    r.write_frame(frame)
    r.stop()
    assert r.recording is False
    if os.path.exists("/tmp/test_recording_write.mp4"):
        os.remove("/tmp/test_recording_write.mp4")


def test_recorder_write_without_start():
    r = Recorder()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    r.write_frame(frame)
    assert r.recording is False


def test_recorder_resize_on_write():
    r = Recorder(output_path="/tmp/test_recording_resize.mp4", width=640, height=480)
    r.start()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    r.write_frame(frame)
    r.stop()
    if os.path.exists("/tmp/test_recording_resize.mp4"):
        os.remove("/tmp/test_recording_resize.mp4")


def test_recorder_double_start():
    r = Recorder(output_path="/tmp/test_recording_double.mp4")
    r.start()
    r.start()
    r.stop()
    if os.path.exists("/tmp/test_recording_double.mp4"):
        os.remove("/tmp/test_recording_double.mp4")


def test_recorder_double_stop():
    r = Recorder(output_path="/tmp/test_recording_double_stop.mp4")
    r.start()
    r.stop()
    r.stop()
    if os.path.exists("/tmp/test_recording_double_stop.mp4"):
        os.remove("/tmp/test_recording_double_stop.mp4")


if __name__ == "__main__":
    test_recorder_init()
    test_recorder_start_stop()
    test_recorder_write_frame()
    test_recorder_write_without_start()
    test_recorder_resize_on_write()
    test_recorder_double_start()
    test_recorder_double_stop()
    print("\nAll recorder tests passed!")
