import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    source: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    processing_width: int = 640
    processing_height: int = 480


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    timeout: float = 0.1


@dataclass
class ModelsConfig:
    gaze_custom: Optional[str] = "models/gaze_cnn.pth"
    gesture_svm: Optional[str] = "models/gesture_svm.pkl"
    gesture_scaler: Optional[str] = "models/gesture_scaler.pkl"
    face_svm: Optional[str] = "models/face_svm.pkl"
    face_scaler: Optional[str] = "models/face_scaler.pkl"


@dataclass
class FaceDetectionConfig:
    min_confidence: float = 0.5
    max_lost_frames: int = 5
    iou_threshold: float = 0.3


@dataclass
class PIDAxisConfig:
    Kp: float = 2.0
    Ki: float = 0.05
    Kd: float = 0.5
    output_limits: list = field(default_factory=lambda: [-30, 30])
    integral_limit: float = 10.0


@dataclass
class PIDConfig:
    pan: PIDAxisConfig = field(default_factory=PIDAxisConfig)
    tilt: PIDAxisConfig = field(default_factory=PIDAxisConfig)
    dead_zone: float = 0.05
    max_angle_delta: float = 5.0


@dataclass
class GestureConfig:
    min_confidence: float = 0.6
    method: str = "svm"


@dataclass
class GazeConfig:
    method: str = "l2cs_net"
    cascade: bool = True


@dataclass
class StateMachineConfig:
    idle_timeout_frames: int = 150


@dataclass
class CalibrationConfig:
    checkerboard_cols: int = 9
    checkerboard_rows: int = 6
    square_size_mm: float = 25.0


@dataclass
class GimbalConfig:
    pan_min: int = 0
    pan_max: int = 180
    tilt_min: int = 45
    tilt_max: int = 135
    pan_center: int = 90
    tilt_center: int = 90
    imu_enabled: bool = True


@dataclass
class IMUConfig:
    enabled: bool = True
    i2c_addr: int = 0x68
    filter_alpha: float = 0.2


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)


def load_config(path: str = "config/default.yaml") -> Config:
    if not os.path.exists(path):
        print(f"Config not found at {path}, using defaults")
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if "camera" in raw:
        cfg.camera = CameraConfig(**raw["camera"])
    if "serial" in raw:
        cfg.serial = SerialConfig(**raw["serial"])
    if "models" in raw:
        cfg.models = ModelsConfig(**raw["models"])
    if "face_detection" in raw:
        cfg.face_detection = FaceDetectionConfig(**raw["face_detection"])
    if "pid" in raw:
        pid = raw["pid"]
        if "pan" in pid:
            cfg.pid.pan = PIDAxisConfig(**pid["pan"])
        if "tilt" in pid:
            cfg.pid.tilt = PIDAxisConfig(**pid["tilt"])
        if "dead_zone" in pid:
            cfg.pid.dead_zone = pid["dead_zone"]
        if "max_angle_delta" in pid:
            cfg.pid.max_angle_delta = pid["max_angle_delta"]
    if "gesture" in raw:
        cfg.gesture = GestureConfig(**raw["gesture"])
    if "gaze" in raw:
        cfg.gaze = GazeConfig(**raw["gaze"])
    if "state_machine" in raw:
        cfg.state_machine = StateMachineConfig(**raw["state_machine"])
    if "calibration" in raw:
        cfg.calibration = CalibrationConfig(**raw["calibration"])
    if "gimbal" in raw:
        cfg.gimbal = GimbalConfig(**raw["gimbal"])
    if "imu" in raw:
        cfg.imu = IMUConfig(**raw["imu"])

    return cfg
