import os
import cv2
import numpy as np
import json
from src.cv.face_detector import FaceDetector


def evaluate_fddb(fddb_dir: str, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    detector = FaceDetector(min_confidence=0.5)

    results = {
        "total_images": 0,
        "faces_detected": 0,
        "inference_times_ms": [],
    }

    if not os.path.exists(fddb_dir):
        print(f"FDDB directory not found: {fddb_dir}")
        print("Skipping face evaluation.")
        return None

    for root, dirs, files in os.walk(fddb_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(root, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            results["total_images"] += 1
            start = cv2.getTickCount()
            faces = detector.detect(img)
            elapsed_ms = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
            results["inference_times_ms"].append(elapsed_ms)
            if faces:
                results["faces_detected"] += 1

    if results["total_images"] > 0:
        results["detection_rate"] = (results["faces_detected"] /
                                      results["total_images"])
        results["avg_inference_time_ms"] = np.mean(results["inference_times_ms"])
        results["std_inference_time_ms"] = np.std(results["inference_times_ms"])

    report_path = os.path.join(output_dir, "logs", "face_evaluation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Face evaluation saved to {report_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/face/fddb")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    evaluate_fddb(args.data, args.output)
