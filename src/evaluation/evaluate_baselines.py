"""Baseline comparison: run OpenCV Haar Cascade + MediaPipe on WIDER Face val.
Compares detection rate, FPS, and accuracy against our custom FaceCNN.

Usage: PYTHONPATH="." python src/evaluation/evaluate_baselines.py
"""
import os, sys, json, time, glob
import cv2
import numpy as np
from src.cv.face_detector_cnn import FaceCNN

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data/face/widerface/WIDER_val/images")
OUT = os.path.join(REPO, "reports/logs")
os.makedirs(OUT, exist_ok=True)


def get_test_images(max_images=200):
    paths = sorted(glob.glob(os.path.join(DATA, "*/*.jpg")))[:max_images]
    print(f"Using {len(paths)} test images from {DATA}")
    return paths


def benchmark_haar(images):
    """OpenCV Haar Cascade face detector."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    total = 0
    detected = 0
    times = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        total += 1
        if len(faces) > 0:
            detected += 1
    return {
        "detection_rate": detected / max(total, 1),
        "avg_ms": float(np.mean(times)),
        "fps": total / max(sum(times) / 1000, 1e-6),
        "total_images": total,
        "faces_found": detected,
    }


def benchmark_mediapipe(images):
    """MediaPipe face detection (if available)."""
    try:
        import mediapipe as mp
        mp_fd = mp.solutions.face_detection
        detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    except ImportError:
        return {"error": "mediapipe not installed"}

    total = 0
    detected = 0
    times = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        results = detector.process(rgb)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        total += 1
        if results.detections:
            detected += 1
    return {
        "detection_rate": detected / max(total, 1),
        "avg_ms": float(np.mean(times)),
        "fps": total / max(sum(times) / 1000, 1e-6),
        "total_images": total,
        "faces_found": detected,
    }


def benchmark_custom_cnn(images, model_path):
    """Our custom FaceCNN."""
    detector = FaceCNN(model_path=model_path, confidence_threshold=0.25)
    total = 0
    detected = 0
    times = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        t0 = time.perf_counter()
        faces = detector.detect(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        total += 1
        if faces:
            detected += 1
    return {
        "detection_rate": detected / max(total, 1),
        "avg_ms": float(np.mean(times)),
        "fps": total / max(sum(times) / 1000, 1e-6),
        "total_images": total,
        "faces_found": detected,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/face_cnn_best.pth")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--skip-haar", action="store_true")
    parser.add_argument("--skip-mediapipe", action="store_true")
    args = parser.parse_args()

    images = get_test_images(args.max_images)
    results = {}

    if not args.skip_haar:
        print("\nBenchmarking Haar Cascade...")
        results["haar"] = benchmark_haar(images)
        print(f"  Detection rate: {results['haar']['detection_rate']:.3f}, "
              f"{results['haar']['avg_ms']:.0f}ms, {results['haar']['fps']:.0f} FPS")

    if not args.skip_mediapipe:
        print("\nBenchmarking MediaPipe...")
        results["mediapipe"] = benchmark_mediapipe(images)
        if "error" in results["mediapipe"]:
            print(f"  {results['mediapipe']['error']}")
        else:
            print(f"  Detection rate: {results['mediapipe']['detection_rate']:.3f}, "
                  f"{results['mediapipe']['avg_ms']:.0f}ms, {results['mediapipe']['fps']:.0f} FPS")

    print(f"\nBenchmarking Custom FaceCNN (epoch 32)...")
    results["custom_cnn"] = benchmark_custom_cnn(images, args.model)
    print(f"  Detection rate: {results['custom_cnn']['detection_rate']:.3f}, "
          f"{results['custom_cnn']['avg_ms']:.0f}ms, {results['custom_cnn']['fps']:.0f} FPS")

    # Save
    out_path = os.path.join(OUT, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("Comparison Summary")
    print(f"{'='*50}")
    print(f"{'Method':20s} {'Det Rate':>10s} {'Avg ms':>10s} {'FPS':>10s}")
    print(f"{'-'*50}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:20s} {'ERROR':>10s}")
        else:
            print(f"{name:20s} {r['detection_rate']:>10.3f} {r['avg_ms']:>9.0f}ms {r['fps']:>9.0f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
