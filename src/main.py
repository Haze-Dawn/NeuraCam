import cv2
import json
import time
import os
import signal
import collections
from collections import deque
import numpy as np
from datetime import datetime

_has_display = bool(os.environ.get("DISPLAY")) or os.name == "nt"
if _has_display:
    try:
        import cv2
        cv2.namedWindow("probe")
        cv2.destroyWindow("probe")
    except Exception:
        _has_display = False
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
from src.cv.face_detector_cnn import FaceCNN, FaceCNNv5, FaceCNNv7_1
from src.cv.face_cnn_v0 import FaceCNNV0Wrapper
from src.cv.face_tracker import KalmanTracker, Face, BoundingBox
from src.cv.gesture_classifier import GestureClassifier, HandDetector
from src.control.gimbal import GimbalController
from src.control.pid import PIDController, compute_adaptive_dead_zone
from src.control.state_machine import StateMachine, Mode
from src.utils.visualization import (
    draw_debug_overlay, compute_framing_error
)
from src.utils.ipc_server import IPCServer
from src.utils.mjpeg_server import MJPEGServer


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

    arch = getattr(cfg.face_detection, 'architecture', 'v4')
    if arch == "v0" and cfg.models.face_cnn_v0 and os.path.exists(cfg.models.face_cnn_v0):
        face_cnn = FaceCNNV0Wrapper(
            model_path=cfg.models.face_cnn_v0,
            onnx_path=cfg.models.face_cnn_v0.replace('.pth', '.onnx'),
            score_threshold=getattr(cfg.face_detection_v0, 'confidence_threshold', 0.3),
            nms_threshold=getattr(cfg.face_detection_v0, 'nms_iou_threshold', 0.45),
            backend=getattr(cfg.face_detection_v0, 'backend', 'pytorch'),
        )
    elif arch == "v71" and cfg.models.face_cnn_v71 and os.path.exists(cfg.models.face_cnn_v71):
        face_cnn = FaceCNNv7_1(
            model_path=cfg.models.face_cnn_v71,
            confidence_threshold=getattr(cfg.face_detection_v71, 'confidence_threshold', 0.25),
            nms_iou_threshold=getattr(cfg.face_detection_v71, 'nms_iou_threshold', 0.3),
        )
    elif arch == "v5" and cfg.models.face_cnn_v5 and os.path.exists(cfg.models.face_cnn_v5):
        face_cnn = FaceCNNv5(
            model_path=cfg.models.face_cnn_v5,
            confidence_threshold=getattr(cfg.face_detection_v5, 'confidence_threshold', 0.3),
            nms_iou_threshold=getattr(cfg.face_detection_v5, 'nms_iou_threshold', 0.3),
        )
    else:
        face_cnn = FaceCNN(
            model_path=cfg.models.face_cnn,
            confidence_threshold=cfg.face_detection.confidence_threshold,
            nms_iou_threshold=cfg.face_detection.nms_iou_threshold,
            input_size=cfg.face_detection.input_size,
            skip_scale_threshold=cfg.face_detection.skip_scale_threshold,
            num_anchors=cfg.face_detection.num_anchors,
            use_depthwise=cfg.face_detection.use_depthwise,
            use_se=cfg.face_detection.use_se,
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
        rf_model_path=cfg.models.gesture_detector,
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
        pan_min=cfg.gimbal.pan_min,
        pan_max=cfg.gimbal.pan_max,
        tilt_min=cfg.gimbal.tilt_min,
        tilt_max=cfg.gimbal.tilt_max,
        center=cfg.gimbal.pan_center,
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

    ipc_server = IPCServer()
    mjpeg_server = MJPEGServer()
    mjpeg_server.start()
    events = deque(maxlen=20)

    def add_event(msg):
        events.append(msg)
        ipc_server.add_event(msg)

    logger = ExperimentLogger()
    profiler = LatencyProfiler(window=30)

    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    current_fps = 0.0
    running = True
    last_frame_time = time.time()

    time.sleep(1.0)

    add_event("System started")
    print("System ready. Controls: q=quit, h=home, space=lock, r=record, "
          "wave=hand tracking, double-wave=toggle zoom")

    face = None
    prev_mode = state_machine.mode.value

    def _choose_scales():
        """Scale-skipping: use 1.0x when tracking, escalate when searching.
        Average cost: ~1.03 forward passes per frame."""
        if face and kalman_tracker.lost_frames == 0:
            return [1.0], {"p4": 0.30, "p2": 0.20}  # locked: fast path
        elif kalman_tracker.lost_frames < 10:
            return [1.0], {"p4": 0.15, "p2": 0.10}  # recent coast: low thresh
        else:
            return [1.0, 1.5, 2.0], {"p4": 0.15, "p2": 0.10}  # SEARCH: escalate

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
        scales, conf_thresh = _choose_scales()
        faces = face_cnn.detect(processing_frame, conf_thresholds=conf_thresh, scales=scales)
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
            gimbal.flush()
            tol = 2
            if (abs(gimbal.pan_angle - gimbal.center) <= tol
                    and abs(gimbal.tilt_angle - gimbal.center) <= tol):
                if face:
                    state_machine.mode = Mode.TRACKING
                else:
                    state_machine.mode = Mode.SEARCH

        # Hand detection: runs every frame for wave detection,
        # and every frame when tracking hand or zoom is active
        need_hand_detection = (
            frame_count % 3 == 0
            or state_machine.mode in (Mode.TRACKING_HAND, Mode.LOCKED)
            or state_machine.zoom_mode_active
        )
        hand_roi = None
        hand_bbox = None
        hand_detected = False
        if need_hand_detection:
            hand_result = hand_detector.detect(
                frame, face_bbox=face.bbox if face else None
            )
            if hand_result:
                hand_roi, hand_bbox = hand_result
                hand_detected = True
                state_machine.update_hand_status(True)
            else:
                state_machine.update_hand_status(False)

        # Wave detection: runs whenever hand is visible
        if hand_detected:
            wave_type = hand_detector.detect_wave(hand_bbox, w, frame_count)
            if wave_type:
                state_machine.process_wave(wave_type)
                if wave_type == 'single':
                    print("Wave detected: toggled hand tracking")
                elif wave_type == 'double':
                    print("Double wave detected: toggled zoom mode")

        # Squeeze zoom: runs every frame when zoom is active and hand is visible
        if state_machine.zoom_mode_active and hand_roi is not None:
            defect_count = gesture_classifier.compute_defect_count(hand_roi)
            state_machine.update_hand_zoom(defect_count)

        # Track PID errors and outputs (for IPC state)
        error_x = 0.0
        error_y = 0.0
        delta_pan = 0.0
        delta_tilt = 0.0

        # Face tracking PID
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
            gimbal.flush()
            profiler.mark("pid")

        # Hand tracking PID
        elif state_machine.mode == Mode.TRACKING_HAND and hand_detected:
            profiler.mark("pid")
            if cfg.pid.dead_zone_adaptive:
                dead_zone = compute_adaptive_dead_zone(
                    BoundingBox(x=hand_bbox[0], y=hand_bbox[1],
                                w=hand_bbox[2], h=hand_bbox[3]),
                    (w, h)
                )
            else:
                dead_zone = cfg.pid.dead_zone_base
            error_x, error_y = compute_framing_error(
                BoundingBox(x=hand_bbox[0], y=hand_bbox[1],
                            w=hand_bbox[2], h=hand_bbox[3]),
                (w, h), dead_zone
            )
            delta_pan = pid_pan.update(error_x, frame_dt)
            delta_tilt = pid_tilt.update(error_y, frame_dt)
            gimbal.set_pan_delta(delta_pan)
            gimbal.set_tilt_delta(delta_tilt)
            gimbal.flush()
            profiler.mark("pid")

        elif state_machine.mode == Mode.IDLE:
            pid_pan.reset()
            pid_tilt.reset()

        elif state_machine.mode == Mode.SEARCH:
            sweep_progress = state_machine.search_progress
            target_pan = int(90 + (sweep_progress - 0.5) * 180)
            target_pan = max(0, min(180, target_pan))
            gimbal.set_pan(target_pan)
            gimbal.flush()
            pid_pan.reset()
            pid_tilt.reset()

        # Gesture classification every 6th frame
        gesture_result = None
        if frame_count % 6 == 0 and state_machine.mode in (
                Mode.TRACKING, Mode.TRACKING_HAND, Mode.LOCKED):
            profiler.mark("gesture")
            if hand_roi is not None:
                gesture_result = gesture_classifier.predict(hand_roi)
                state_machine.process_gesture(gesture_result.gesture)
            profiler.mark("gesture")

        # Apply digital zoom for display
        display_frame = frame
        if state_machine.zoom_level > 1.0:
            zoom = state_machine.zoom_level
            new_w = int(w / zoom)
            new_h = int(h / zoom)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            display_frame = frame[y1:y1+new_h, x1:x1+new_w]
            display_frame = cv2.resize(display_frame, (w, h))

        profiler.mark("display")
        overlay = draw_debug_overlay(
            frame=display_frame,
            face=face,
            gesture=gesture_result,
            mode=state_machine.mode,
            fps=current_fps,
            recording=recorder.recording,
            gimbal_angles=(gimbal.pan_angle, gimbal.tilt_angle),
            imu_angles=(gimbal.imu_pitch, gimbal.imu_roll, gimbal.imu_yaw),
            kalman_uncertainty=kalman_tracker.uncertainty,
            zoom_level=state_machine.zoom_level,
            tracking_target=state_machine.tracking_target,
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
                "zoom_level": state_machine.zoom_level,
                "tracking_target": state_machine.tracking_target,
                "latency_ms": latency,
            }
            logger.log(log_entry)

        # Track mode transitions for event log
        current_mode = state_machine.mode.value
        if current_mode != prev_mode:
            add_event(f"Mode: {prev_mode} → {current_mode}")
            prev_mode = current_mode

        # IPC: send state + frame to Rust clients
        profiler.mark("ipc")
        latency = profiler.snapshot()
        ipc_state = {
            "fps": round(current_fps, 1),
            "mode": state_machine.mode.value,
            "tracking_target": state_machine.tracking_target,
            "face_detected": face is not None,
            "face_x": face.bbox.center_x if face else 0,
            "face_y": face.bbox.center_y if face else 0,
            "face_w": face.bbox.w if face else 0,
            "face_h": face.bbox.h if face else 0,
            "face_confidence": round(face.confidence, 3) if face else 0,
            "frame_w": w,
            "frame_h": h,
            "pan_angle": gimbal.pan_angle,
            "tilt_angle": gimbal.tilt_angle,
            "pan_target": gimbal._pending_pan if hasattr(gimbal, '_pending_pan') else gimbal.pan_angle,
            "tilt_target": gimbal._pending_tilt if hasattr(gimbal, '_pending_tilt') else gimbal.tilt_angle,
            "imu_pitch": round(gimbal.imu_pitch, 1),
            "imu_roll": round(gimbal.imu_roll, 1),
            "imu_yaw": round(gimbal.imu_yaw, 1),
            "kalman_uncertainty": round(kalman_tracker.uncertainty, 3),
            "zoom_level": state_machine.zoom_level,
            "gesture": gesture_result.gesture if gesture_result else "NONE",
            "gesture_method": gesture_result.method if gesture_result else "",
            "recording": recorder.recording,
            "serial_connected": gimbal._connected,
            "hand_detected": hand_detected,
            "pid_pan_error": round(error_x, 3),
            "pid_tilt_error": round(error_y, 3),
            "pid_pan_output": round(delta_pan, 3),
            "pid_tilt_output": round(delta_tilt, 3),
            "pid_pan_p": round(pid_pan.Kp * error_x, 3),
            "pid_pan_i": round(pid_pan.Ki * pid_pan.integral, 3),
            "pid_pan_d": round(pid_pan.Kd * pid_pan.filtered_derivative, 3),
            "pid_tilt_p": round(pid_tilt.Kp * error_y, 3),
            "pid_tilt_i": round(pid_tilt.Ki * pid_tilt.integral, 3),
            "pid_tilt_d": round(pid_tilt.Kd * pid_tilt.filtered_derivative, 3),
            "latency_ms": latency,
            "events": list(events),
            "timestamp": time.time(),
        }
        try:
            _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg.tobytes()
            mjpeg_server.publish(jpeg_bytes)
            ipc_server.send_state(ipc_state, jpeg_bytes)
        except Exception:
            ipc_server.send_state(ipc_state)
        profiler.mark("ipc")

        if _has_display:
            cv2.imshow("NeuraCam", overlay)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 0xFF
        profiler.mark("display")

        # Handle IPC keyboard input (from Rust GUI)
        ipc_key = ipc_server.read_key()
        if ipc_key:
            key = ord(ipc_key)

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
    gimbal.close()
    camera.release()
    recorder.stop()
    mjpeg_server.stop()
    ipc_server.stop()
    logger.save("session")
    if _has_display:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print("System stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)
