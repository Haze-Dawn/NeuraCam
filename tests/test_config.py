import os
import tempfile
import yaml
from src.utils.config import (
    load_config, Config, CameraConfig, FaceDetectionConfig,
    PIDConfig, PIDAxisConfig, SerialConfig, GimbalConfig,
    KalmanConfig, GestureConfig, HandDetectionConfig,
    StateMachineConfig, CalibrationConfig, ModelsConfig
)


def test_load_config_defaults():
    """Loading a nonexistent path returns defaults."""
    cfg = load_config("/nonexistent/config.yaml")
    assert isinstance(cfg, Config)
    assert cfg.camera.source == 0
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.camera.fps == 30
    assert cfg.camera.processing_width == 640
    assert cfg.camera.processing_height == 480


def test_load_config_custom_yaml():
    data = {
        "camera": {"source": 1, "width": 1920, "height": 1080},
        "serial": {"port": "COM5", "baud": 9600},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    cfg = load_config(path)
    assert cfg.camera.source == 1
    assert cfg.camera.width == 1920
    assert cfg.camera.height == 1080
    assert cfg.serial.port == "COM5"
    assert cfg.serial.baud == 9600
    os.unlink(path)


def test_load_config_partial():
    """Missing keys fall back to defaults."""
    data = {"camera": {"source": 2}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    cfg = load_config(path)
    assert cfg.camera.source == 2
    assert cfg.camera.width == 1280
    assert cfg.serial.baud == 115200
    os.unlink(path)


def test_load_config_empty_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({}, f)
        path = f.name
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    os.unlink(path)


def test_camera_config_defaults():
    c = CameraConfig()
    assert c.source == 0
    assert c.width == 1280
    assert c.height == 720
    assert c.fps == 30
    assert c.processing_width == 640
    assert c.processing_height == 480
    assert c.use_capture_thread is True


def test_face_detection_config_defaults():
    f = FaceDetectionConfig()
    assert f.confidence_threshold == 0.3
    assert f.nms_iou_threshold == 0.25
    assert f.input_size == 128
    assert f.skip_scale_threshold == 0.9


def test_pid_config_hierarchy():
    p = PIDConfig()
    assert p.pan.Kp == 2.0
    assert p.pan.Ki == 0.05
    assert p.pan.Kd == 0.5
    assert p.pan.output_limits == [-30, 30]
    assert p.pan.integral_limit == 10.0
    assert p.tilt.Kp == 2.0
    assert p.dead_zone_base == 0.05
    assert p.dead_zone_adaptive is True
    assert p.max_angle_delta == 5.0


def test_serial_config():
    s = SerialConfig()
    assert s.port == "auto"
    assert s.baud == 115200
    assert s.timeout == 0.1
    assert s.batch_commands is True
    assert s.control_rate_hz == 100.0


def test_gimbal_config():
    g = GimbalConfig()
    assert g.pan_min == 0
    assert g.pan_max == 180
    assert g.tilt_min == 45
    assert g.tilt_max == 135
    assert g.pan_center == 90
    assert g.tilt_center == 90


def test_kalman_config():
    k = KalmanConfig()
    assert k.process_noise == 0.01
    assert k.measurement_noise == 0.1
    assert k.max_lost_frames == 5
    assert k.iou_threshold == 0.3


def test_gesture_config():
    g = GestureConfig()
    assert g.min_confidence == 0.6
    assert g.pca_components == 80


def test_hand_detection_config():
    h = HandDetectionConfig()
    assert h.ycrcb_lower == [0, 133, 77]
    assert h.ycrcb_upper == [255, 173, 127]
    assert h.use_motion is True
    assert h.min_area == 1000


def test_state_machine_config():
    s = StateMachineConfig()
    assert s.idle_timeout_frames == 150
    assert s.gesture_hold_frames == 5
    assert s.search_duration == 90


def test_calibration_config():
    c = CalibrationConfig()
    assert c.checkerboard_cols == 9
    assert c.checkerboard_rows == 6
    assert c.square_size_mm == 25.0


def test_models_config():
    m = ModelsConfig()
    assert m.face_cnn == "models/face_cnn.pth"
    assert m.gesture_svm == "models/gesture_svm.pkl"
    assert m.gesture_pca == "models/gesture_pca.pkl"
    assert m.gesture_scaler == "models/gesture_scaler.pkl"


def test_pid_axis_config():
    p = PIDAxisConfig()
    assert p.Kp == 2.0
    assert p.Ki == 0.05
    assert p.Kd == 0.5
    assert p.output_limits == [-30, 30]
    assert p.integral_limit == 10.0


if __name__ == "__main__":
    test_load_config_defaults()
    test_load_config_custom_yaml()
    test_load_config_partial()
    test_load_config_empty_yaml()
    test_camera_config_defaults()
    test_face_detection_config_defaults()
    test_pid_config_hierarchy()
    test_serial_config()
    test_gimbal_config()
    test_kalman_config()
    test_gesture_config()
    test_hand_detection_config()
    test_state_machine_config()
    test_calibration_config()
    test_models_config()
    test_pid_axis_config()
    print("\nAll config tests passed!")
