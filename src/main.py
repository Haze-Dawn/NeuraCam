import cv2
import json
import time
import os
from datetime import datetime

from src.utils.config import load_config, Config
from src.capture.camera import Camera
from src.capture.recorder import Recorder
from src.cv.face_detector import FaceDetector, FaceTracker
from src.cv.gaze_estimator import GazeEstimator
from src.cv.gesture_classifier import GestureClassifier
from src.control.gimbal import GimbalController
from src.control.pid import PIDController
from src.control.state_machine import StateMachine, Mode
from src.utils.visualization import (
    draw_debug_overlay, compute_framing_error, crop_face_region
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


def run(cfg: Config = None):
    if cfg is None:
        cfg = load_config()

    camera = Camera(
        source=cfg.camera.source,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        processing_width=cfg.camera.processing_width,
        processing_height=cfg.camera.processing_height
    )

    face_detector = FaceDetector(
        model_path=cfg.models.face_svm,
        scaler_path=cfg.models.face_scaler,
        min_confidence=cfg.face_detection.min_confidence
    )
    face_tracker = FaceTracker(
        max_lost_frames=cfg.face_detection.max_lost_frames,
        iou_threshold=cfg.face_detection.iou_threshold
    )

    gaze_estimator = GazeEstimator(
        custom_cnn_path=cfg.models.gaze_custom
    )

    gesture_classifier = GestureClassifier(
        svm_path=cfg.models.gesture_svm,
        scaler_path=cfg.models.gesture_scaler,
        min_confidence=cfg.gesture.min_confidence
    )

    gimbal = GimbalController(
        port=cfg.serial.port,
        baud=cfg.serial.baud,
        timeout=cfg.serial.timeout
    )
    gimbal.home()

    pid_pan = PIDController(
        Kp=cfg.pid.pan.Kp, Ki=cfg.pid.pan.Ki, Kd=cfg.pid.pan.Kd,
        output_limits=tuple(cfg.pid.pan.output_limits),
        integral_limit=cfg.pid.pan.integral_limit
    )
    pid_tilt = PIDController(
        Kp=cfg.pid.tilt.Kp, Ki=cfg.pid.tilt.Ki, Kd=cfg.pid.tilt.Kd,
        output_limits=tuple(cfg.pid.tilt.output_limits),
        integral_limit=cfg.pid.tilt.integral_limit
    )

    state_machine = StateMachine(
        idle_timeout_frames=cfg.state_machine.idle_timeout_frames
    )

    recorder = Recorder(
        fps=cfg.camera.fps,
        width=cfg.camera.width,
        height=cfg.camera.height
    )

    logger = ExperimentLogger()

    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    current_fps = 0.0
    running = True
    homing = False
    last_frame_time = time.time()

    time.sleep(1.0)

    print("System ready. Controls: q=quit, h=home, space=lock, r=record")

    while running:
        frame_obj = camera.read()
        if frame_obj is None:
            continue

        frame = frame_obj.data
        frame_count += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        now = time.time()
        frame_dt = now - last_frame_time
        last_frame_time = now
        h, w = frame.shape[:2]

        processing_frame = camera.get_processing_frame(frame)
        faces = face_detector.detect(processing_frame)
        face = face_tracker.update(faces)
        if face:
            scale_x = w / camera.processing_width
            scale_y = h / camera.processing_height
            face.bbox.x = int(face.bbox.x * scale_x)
            face.bbox.y = int(face.bbox.y * scale_y)
            face.bbox.w = int(face.bbox.w * scale_x)
            face.bbox.h = int(face.bbox.h * scale_y)
        state_machine.update_face_status(face is not None)

        if state_machine.mode == Mode.HOME and not homing:
            gimbal.home()
            homing = True

        if homing:
            if face:
                state_machine.finish_homing()
                homing = False
            else:
                state_machine.finish_homing()
                homing = False

        if state_machine.mode == Mode.TRACKING and face:
            error_x, error_y = compute_framing_error(
                face.bbox, (w, h), cfg.pid.dead_zone
            )
            delta_pan = pid_pan.update(error_x, frame_dt)
            delta_tilt = pid_tilt.update(error_y, frame_dt)
            gimbal.set_pan_delta(delta_pan)
            gimbal.set_tilt_delta(delta_tilt)

        elif state_machine.mode == Mode.IDLE:
            pid_pan.reset()
            pid_tilt.reset()

        gesture_result = None
        gaze_result = None

        if frame_count % 6 == 0:
            gesture_result = gesture_classifier.predict(frame)
            if gesture_result:
                state_machine.process_gesture(gesture_result.gesture)

            if face:
                face_crop = crop_face_region(frame, face.bbox)
                gaze_result = gaze_estimator.predict(face_crop)
                if gaze_result:
                    state_machine.update_gaze(gaze_result.direction)

        overlay = draw_debug_overlay(
            frame=frame,
            face=face,
            gaze=gaze_result,
            gesture=gesture_result,
            mode=state_machine.mode,
            fps=current_fps,
            recording=recorder.recording,
            gimbal_angles=(gimbal.pan_angle, gimbal.tilt_angle),
            imu_angles=(gimbal.imu_pitch, gimbal.imu_roll, gimbal.imu_yaw),
        )

        recorder.write_frame(overlay)

        if frame_count % 30 == 0:
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
                "gaze": gaze_result.direction_label if gaze_result else None,
            }
            logger.log(log_entry)

        cv2.imshow("AI Gimbal Camera", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            running = False
        elif key == ord('h'):
            gimbal.home()
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

    gimbal.home()
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
