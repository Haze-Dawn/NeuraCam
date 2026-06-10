import os
import yaml
from dataclasses import dataclass, field, fields
from typing import Optional, List


@dataclass
class CameraConfig:
    source: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    processing_width: int = 640
    processing_height: int = 480
    use_capture_thread: bool = True


@dataclass
class SerialConfig:
    port: str = "auto"
    baud: int = 115200
    timeout: float = 0.1
    batch_commands: bool = True
    control_rate_hz: float = 100.0


@dataclass
class ModelsConfig:
    face_cnn: Optional[str] = "models/face_cnn.pth"
    face_cnn_v5: Optional[str] = "models/face_cnn_v5_best.pth"
    face_cnn_v71: Optional[str] = "models/face_cnn_v71_best.pth"
    face_cnn_v0: Optional[str] = "models/face_cnn_v0/face_cnn_v0.pth"
    gesture_detector: Optional[str] = "models/gesture_detector_rf.pkl"
    gesture_pca: Optional[str] = "models/gesture_pca.pkl"
    gesture_svm: Optional[str] = "models/gesture_svm.pkl"
    gesture_scaler: Optional[str] = "models/gesture_scaler.pkl"


@dataclass
class FaceDetectionConfig:
    architecture: str = "v4"
    confidence_threshold: float = 0.3
    nms_iou_threshold: float = 0.25
    input_size: int = 128
    skip_scale_threshold: float = 0.9
    num_anchors: int = 3
    use_depthwise: bool = True
    use_se: bool = True


@dataclass
class FaceDetectionV5Config:
    confidence_threshold: float = 0.3
    nms_iou_threshold: float = 0.3


@dataclass
class FaceDetectionV7_1Config:
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.3
    resolution_adaptive: bool = True
    use_track_consistency_filter: bool = True


@dataclass
class FaceDetectionV0Config:
    confidence_threshold: float = 0.3
    nms_iou_threshold: float = 0.45
    input_size: int = 640
    backend: str = "pytorch"


@dataclass
class KalmanConfig:
    process_noise: float = 0.01
    measurement_noise: float = 0.1
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
    dead_zone_base: float = 0.05
    dead_zone_adaptive: bool = True
    max_angle_delta: float = 5.0


@dataclass
class GestureConfig:
    min_confidence: float = 0.6
    pca_components: int = 200


@dataclass
class HandDetectionConfig:
    ycrcb_lower: List[int] = field(default_factory=lambda: [0, 133, 77])
    ycrcb_upper: List[int] = field(default_factory=lambda: [255, 173, 127])
    hsv_lower: List[int] = field(default_factory=lambda: [0, 30, 60])
    hsv_upper: List[int] = field(default_factory=lambda: [20, 150, 255])
    use_motion: bool = True
    min_area: int = 1000


@dataclass
class StateMachineConfig:
    idle_timeout_frames: int = 150
    gesture_hold_frames: int = 5
    search_duration: int = 90


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


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    face_detection_v5: FaceDetectionV5Config = field(default_factory=FaceDetectionV5Config)
    face_detection_v71: FaceDetectionV7_1Config = field(default_factory=FaceDetectionV7_1Config)
    face_detection_v0: FaceDetectionV0Config = field(default_factory=FaceDetectionV0Config)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    hand_detection: HandDetectionConfig = field(default_factory=HandDetectionConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)


def load_config(path: str = "config/default.yaml") -> Config:
    if not os.path.exists(path):
        print(f"Config not found at {path}, using defaults")
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if raw is None:
        return cfg

    if "camera" in raw:
        cfg.camera = CameraConfig(**raw["camera"])
    if "serial" in raw:
        cfg.serial = SerialConfig(**raw["serial"])
    if "models" in raw:
        model_fields = {f.name for f in fields(ModelsConfig)}
        filtered = {k: v for k, v in raw["models"].items() if k in model_fields}
        extra = set(raw["models"].keys()) - model_fields
        if extra:
            import warnings
            warnings.warn(f"Ignoring unknown model config keys: {sorted(extra)}")
        cfg.models = ModelsConfig(**filtered)
    if "face_detection" in raw:
        fd_fields = {f.name for f in fields(FaceDetectionConfig)}
        filtered = {k: v for k, v in raw["face_detection"].items() if k in fd_fields}
        extra = set(raw["face_detection"].keys()) - fd_fields
        if extra:
            import warnings
            warnings.warn(f"Ignoring unknown face_detection keys: {sorted(extra)}")
        cfg.face_detection = FaceDetectionConfig(**filtered)
    if "face_detection_v5" in raw:
        cfg.face_detection_v5 = FaceDetectionV5Config(**raw["face_detection_v5"])
    if "face_detection_v71" in raw:
        cfg.face_detection_v71 = FaceDetectionV7_1Config(**raw["face_detection_v71"])
    if "face_detection_v0" in raw:
        cfg.face_detection_v0 = FaceDetectionV0Config(**raw["face_detection_v0"])
    if "kalman" in raw:
        cfg.kalman = KalmanConfig(**raw["kalman"])
    if "pid" in raw:
        pid = raw["pid"]
        if "pan" in pid:
            cfg.pid.pan = PIDAxisConfig(**pid["pan"])
        if "tilt" in pid:
            cfg.pid.tilt = PIDAxisConfig(**pid["tilt"])
        if "dead_zone_base" in pid:
            cfg.pid.dead_zone_base = pid["dead_zone_base"]
        if "dead_zone_adaptive" in pid:
            cfg.pid.dead_zone_adaptive = pid["dead_zone_adaptive"]
        if "max_angle_delta" in pid:
            cfg.pid.max_angle_delta = pid["max_angle_delta"]
    if "gesture" in raw:
        cfg.gesture = GestureConfig(**raw["gesture"])
    if "hand_detection" in raw:
        cfg.hand_detection = HandDetectionConfig(**raw["hand_detection"])
    if "state_machine" in raw:
        cfg.state_machine = StateMachineConfig(**raw["state_machine"])
    if "calibration" in raw:
        cfg.calibration = CalibrationConfig(**raw["calibration"])
    if "gimbal" in raw:
        cfg.gimbal = GimbalConfig(**raw["gimbal"])

    return cfg
