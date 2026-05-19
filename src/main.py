import cv2
import json
import time
import os
import signal
import collections
import numpy as np
from datetime import datetime

_has_display = bool(os.environ.get("DISPLAY")) or os.name == "nt"
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\nSignal {signum} received, shutting down...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

from src.utils.config import load_config, Config
from src.capture.camera import Camera
from src.capture.recorder import Recorder
from src.cv.face_detector_cnn import FaceCNN
from src.cv.face_tracker import KalmanTracker, Face, BoundingBox
from src.cv.gesture_classifier import GestureClassifier, HandDetector
from src.control.gimbal import GimbalController
from src.control.pid import PIDController, compute_adaptive_dead_zone
from src.control.state_machine import StateMachine, Mode
from src.utils.visualization import (
    draw_debug_overlay, compute_framing_error
)


class ExperimentLogger:
    def __init__(self, log_dir: str = "experiments"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.entries = []

    def log(self, entry: dict):
        self.entries.append(entry)

    def save(self, name: str = "session"):
        path = os.path.join(
            self.log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)
        print(f"Log saved to {path}")


class LatencyProfiler:
    """Per-component latency waterfall tracker.
    Records rolling average of each main-loop component's execution time.
    """
    def __init__(self, window=30):
        self.window = window
        self.times = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self._marks = {}

    def mark(self, name: str):
        """Call mark('start') then mark('detect'), etc. to time intervals."""
        now = time.perf_counter()
        if name in self._marks:
            elapsed = (now - self._marks.pop(name)) * 1000
            self.times[name].append(elapsed)
        else:
            self._marks[name] = now

    def avg(self, name: str) -> float:
        if not self.times[name]:
            return 0.0
        return float(np.mean(self.times[name]))

    def snapshot(self) -> dict:
        """Return dict with avg ms per component."""
        return {k: round(self.avg(k), 2) for k in self.times}

    def total_avg(self) -> float:
        """Total loop time = sum of all timing marks."""
        return sum(self.avg(k) for k in self.times)


def run(cfg: Config = None):
    if cfg is None:
        cfg = load_config()

    camera = Camera(
        source=cfg.camera.source,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        processing_width=cfg.camera.processing_width,
        processing_height=cfg.camera.processing_height,
        use_capture_thread=cfg.camera.use_capture_thread,
    )

    face_cnn = FaceCNN(
        model_path=cfg.models.face_cnn,
        confidence_threshold=cfg.face_detection.confidence_threshold,
        nms_iou_threshold=cfg.face_detection.nms_iou_threshold,
        input_size=cfg.face_detection.input_size,
        skip_scale_threshold=cfg.face_detection.skip_scale_threshold,
    )

    kalman_tracker = KalmanTracker(
        process_noise=cfg.kalman.process_noise,
        measurement_noise=cfg.kalman.measurement_noise,
        max_lost_frames=cfg.kalman.max_lost_frames,
        iou_threshold=cfg.kalman.iou_threshold,
    )

    hand_detector = HandDetector(
        ycrcb_lower=tuple(cfg.hand_detection.ycrcb_lower),
        ycrcb_upper=tuple(cfg.hand_detection.ycrcb_upper),
        hsv_lower=tuple(cfg.hand_detection.hsv_lower),
        hsv_upper=tuple(cfg.hand_detection.hsv_upper),
        min_area=cfg.hand_detection.min_area,
        use_motion=cfg.hand_detection.use_motion,
    )

    gesture_classifier = GestureClassifier(
        svm_path=cfg.models.gesture_svm,
        scaler_path=cfg.models.gesture_scaler,
        pca_path=cfg.models.gesture_pca,
        min_confidence=cfg.gesture.min_confidence,
    )

    gimbal = GimbalController(
        port=cfg.serial.port,
        baud=cfg.serial.baud,
        timeout=cfg.serial.timeout,
        batch_commands=cfg.serial.batch_commands,
        control_rate_hz=cfg.serial.control_rate_hz,
        max_delta=cfg.pid.max_angle_delta,
    )
    gimbal.home()

    pid_pan = PIDController(
        Kp=cfg.pid.pan.Kp, Ki=cfg.pid.pan.Ki, Kd=cfg.pid.pan.Kd,
        output_limits=tuple(cfg.pid.pan.output_limits),
        integral_limit=cfg.pid.pan.integral_limit,
    )
    pid_tilt = PIDController(
        Kp=cfg.pid.tilt.Kp, Ki=cfg.pid.tilt.Ki, Kd=cfg.pid.tilt.Kd,
        output_limits=tuple(cfg.pid.tilt.output_limits),
        integral_limit=cfg.pid.tilt.integral_limit,
    )

    state_machine = StateMachine(
        idle_timeout_frames=cfg.state_machine.idle_timeout_frames,
        gesture_hold_frames=cfg.state_machine.gesture_hold_frames,
        search_duration=cfg.state_machine.search_duration,
    )

    recorder = Recorder(
        fps=cfg.camera.fps,
        width=cfg.camera.width,
        height=cfg.camera.height,
    )

    logger = ExperimentLogger()
    profiler = LatencyProfiler(window=30)

    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    current_fps = 0.0
    running = True
    last_frame_time = time.time()

    time.sleep(1.0)

    print("System ready. Controls: q=quit, h=home, space=lock, r=record")

    while running and not _shutdown_requested:
        profiler.mark("capture")
        frame_obj = camera.read()
        if frame_obj is None:
            continue
        profiler.mark("capture")

        frame = frame_obj.data
        frame_count += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            elapsed = time.time() - fps_timer
            current_fps = fps_counter / elapsed if elapsed > 0 else 0.0
            fps_counter = 0
            fps_timer = time.time()

        now = time.time()
        frame_dt = now - last_frame_time
        last_frame_time = now
        h, w = frame.shape[:2]

        processing_frame = camera.get_processing_frame(frame)
        profiler.mark("detect")
        faces = face_cnn.detect(processing_frame)
        profiler.mark("detect")
        profiler.mark("track")
        face = kalman_tracker.update(faces, frame_dt)
        profiler.mark("track")

        if face:
            scale_x = w / camera.processing_width
            scale_y = h / camera.processing_height
            scaled_bbox = BoundingBox(
                x=int(face.bbox.x * scale_x),
                y=int(face.bbox.y * scale_y),
                w=int(face.bbox.w * scale_x),
                h=int(face.bbox.h * scale_y),
            )
            face = Face(bbox=scaled_bbox, confidence=face.confidence)
        state_machine.update_face_status(face is not None)

        if state_machine.mode == Mode.HOME:
            gimbal.home()
            if gimbal.pan_angle == 90 and gimbal.tilt_angle == 90:
                if face:
                    state_machine.mode = Mode.TRACKING
                else:
                    state_machine.mode = Mode.SEARCH

        if state_machine.mode == Mode.TRACKING and face:
            profiler.mark("pid")
            if cfg.pid.dead_zone_adaptive:
                dead_zone = compute_adaptive_dead_zone(face.bbox, (w, h))
            else:
                dead_zone = cfg.pid.dead_zone_base

            error_x, error_y = compute_framing_error(
                face.bbox, (w, h), dead_zone
            )
            delta_pan = pid_pan.update(error_x, frame_dt)
            delta_tilt = pid_tilt.update(error_y, frame_dt)
            gimbal.set_pan_delta(delta_pan)
            gimbal.set_tilt_delta(delta_tilt)
            profiler.mark("pid")

        elif state_machine.mode == Mode.IDLE:
            pid_pan.reset()
            pid_tilt.reset()

        elif state_machine.mode == Mode.SEARCH:
            sweep_progress = state_machine.search_progress
            target_pan = int(90 + (sweep_progress - 0.5) * 180)
            target_pan = max(0, min(180, target_pan))
            gimbal.set_pan(target_pan)
            pid_pan.reset()
            pid_tilt.reset()

        gesture_result = None

        if frame_count % 6 == 0 and state_machine.mode in (
                Mode.TRACKING, Mode.LOCKED):
            profiler.mark("gesture")
            hand_roi = hand_detector.detect(
                frame, face_bbox=face.bbox if face else None
            )
            if hand_roi:
                roi, _ = hand_roi
                gesture_result = gesture_classifier.predict(roi)
                state_machine.process_gesture(gesture_result.gesture)
            profiler.mark("gesture")

        profiler.mark("display")
        overlay = draw_debug_overlay(
            frame=frame,
            face=face,
            gesture=gesture_result,
            mode=state_machine.mode,
            fps=current_fps,
            recording=recorder.recording,
            gimbal_angles=(gimbal.pan_angle, gimbal.tilt_angle),
            imu_angles=(gimbal.imu_pitch, gimbal.imu_roll, gimbal.imu_yaw),
            kalman_uncertainty=kalman_tracker.uncertainty,
        )

        recorder.write_frame(overlay)

        if frame_count % 30 == 0:
            gimbal.poll_imu()
            latency = profiler.snapshot()
            log_entry = {
                "frame": frame_count,
                "mode": state_machine.mode.value,
                "face_detected": face is not None,
                "pan_angle": gimbal.pan_angle,
                "tilt_angle": gimbal.tilt_angle,
                "imu_pitch": gimbal.imu_pitch,
                "imu_roll": gimbal.imu_roll,
                "imu_yaw": gimbal.imu_yaw,
                "fps": current_fps,
                "gesture": gesture_result.gesture if gesture_result else None,
                "kalman_uncertainty": kalman_tracker.uncertainty,
                "latency_ms": latency,
            }
            logger.log(log_entry)

        if _has_display:
            cv2.imshow("AI Gimbal Camera", overlay)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 0xFF
        profiler.mark("display")

        if key == ord('q'):
            running = False
        elif key == ord('h'):
            state_machine.mode = Mode.HOME
        elif key == ord(' '):
            state_machine.toggle_lock()
        elif key == ord('r'):
            if recorder.recording:
                recorder.stop()
                print("Recording stopped")
            else:
                recorder.start()
                print("Recording started")

    gimbal.home_immediate()
    camera.release()
    recorder.stop()
    logger.save("session")
    cv2.destroyAllWindows()
    print("System stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)
