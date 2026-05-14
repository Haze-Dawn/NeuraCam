import os
import json
import numpy as np
import time
from src.capture.camera import Camera
from src.cv.face_detector import FaceDetector, FaceTracker
from src.utils.visualization import compute_framing_error


def evaluate_system(duration_sec: int = 60, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    camera = Camera(source=0, width=640, height=480)
    face_detector = FaceDetector(min_confidence=0.5)
    face_tracker = FaceTracker(max_lost_frames=5)

    start_time = time.time()
    last_time = start_time
    frame_count = 0
    errors_x = []
    errors_y = []
    face_detected_frames = 0
    total_frames = 0
    frame_times = []

    while time.time() - start_time < duration_sec:
        frame_obj = camera.read()
        if frame_obj is None:
            continue

        frame = frame_obj.data
        total_frames += 1
        t = time.time()

        faces = face_detector.detect(frame)
        face = face_tracker.update(faces)

        if face:
            face_detected_frames += 1
            h, w = frame.shape[:2]
            error_x, error_y = compute_framing_error(
                face.bbox, (w, h), dead_zone=0.0
            )
            errors_x.append(error_x)
            errors_y.append(error_y)

        frame_times.append(time.time() - t)
        frame_count += 1

    camera.release()
    elapsed = time.time() - start_time
    fps = total_frames / elapsed

    results = {
        "duration_sec": elapsed,
        "total_frames": total_frames,
        "fps": float(fps),
        "face_detection_rate": float(face_detected_frames / max(total_frames, 1)),
        "mean_centering_error_x": float(np.mean(np.abs(errors_x))) if errors_x else 0,
        "mean_centering_error_y": float(np.mean(np.abs(errors_y))) if errors_y else 0,
        "mean_inference_time_ms": float(np.mean(frame_times) * 1000),
        "centering_error_std_x": float(np.std(errors_x)) if errors_x else 0,
        "centering_error_std_y": float(np.std(errors_y)) if errors_y else 0,
    }

    report_path = os.path.join(output_dir, "logs", "system_evaluation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"System evaluation saved to {report_path}")
    print(f"FPS: {fps:.1f}, Detection rate: {results['face_detection_rate']:.3f}")
    print(f"Mean centering error: x={results['mean_centering_error_x']:.4f}, "
          f"y={results['mean_centering_error_y']:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    evaluate_system(args.duration, args.output)
