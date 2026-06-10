"""Benchmark and accuracy comparison: PyTorch FaceCNN vs ONNX FP32 vs ONNX INT8.

Compares detection rate, inference speed, and per-image detection agreement
across all three model formats.

Usage:
  PYTHONPATH="." python src/evaluation/evaluate_onnx.py --model models/face_cnn_best.pth --images data/face/widerface/WIDER_val/images
"""
import os, sys, json, glob, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def get_test_images(image_dir, max_images=200):
    """Get list of image paths for evaluation."""
    paths = sorted(glob.glob(os.path.join(image_dir, "*/*.jpg")))[:max_images]
    if not paths:
        paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:max_images]
    print(f"Found {len(paths)} test images from {image_dir}")
    return paths


def benchmark_pytorch(model_path, images):
    """Benchmark original PyTorch FaceCNN."""
    from src.cv.face_detector_cnn import FaceCNN

    detector = FaceCNN(model_path=model_path, confidence_threshold=0.25)
    times = []
    detected = 0
    total = 0

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
        "type": "PyTorch (TorchScript)",
        "avg_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "std_ms": float(np.std(times)),
        "detection_rate": detected / max(total, 1),
        "avg_fps": total / max(sum(times) / 1000, 1e-6),
        "total_images": total,
        "model_path": model_path,
    }


def benchmark_onnx_model(onnx_path, images):
    """Benchmark ONNX model (FP32 or INT8)."""
    from src.utils.onnx_export import ONNXFaceCNN

    detector = ONNXFaceCNN(onnx_path=onnx_path, confidence_threshold=0.25)
    times = []
    detected = 0
    total = 0

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

    model_type = "ONNX INT8" if "int8" in onnx_path else "ONNX FP32"
    return {
        "type": model_type,
        "avg_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "std_ms": float(np.std(times)),
        "detection_rate": detected / max(total, 1),
        "avg_fps": total / max(sum(times) / 1000, 1e-6),
        "total_images": total,
        "model_path": onnx_path,
    }


def compare_detections_pixelwise(pytorch_path, onnx_path, images, max_compare=50):
    """Compare per-image detection outputs between PyTorch and ONNX models.

    Counts how many of the top-1 detections agree (IoU > 0.5) between
    the two backends on the same input images.
    """
    from src.cv.face_detector_cnn import FaceCNN
    from src.utils.onnx_export import ONNXFaceCNN
    from src.cv.face_tracker import compute_iou

    pt_detector = FaceCNN(model_path=pytorch_path, confidence_threshold=0.25)
    onnx_detector = ONNXFaceCNN(onnx_path=onnx_path, confidence_threshold=0.25)

    results = []
    n_compared = 0
    ious = []

    for path in images[:max_compare]:
        img = cv2.imread(path)
        if img is None:
            continue

        faces_pt = pt_detector.detect(img)
        faces_onnx = onnx_detector.detect(img)

        n_compared += 1
        best = faces_pt[0] if faces_pt else None
        best_o = faces_onnx[0] if faces_onnx else None

        if best and best_o:
            iou = compute_iou(best.bbox, best_o.bbox)
            ious.append(iou)
            results.append({
                "image": os.path.basename(path),
                "pt_conf": round(best.confidence, 4),
                "onnx_conf": round(best_o.confidence, 4),
                "iou": round(iou, 4),
                "pt_bbox": [best.bbox.x, best.bbox.y, best.bbox.w, best.bbox.h],
                "onnx_bbox": [best_o.bbox.x, best_o.bbox.y, best_o.bbox.w, best_o.bbox.h],
                "match": iou >= 0.5,
            })
        elif best is None and best_o is None:
            ious.append(1.0)
            results.append({"image": os.path.basename(path), "match": True,
                           "note": "both_no_detection"})
        else:
            ious.append(0.0)
            results.append({"image": os.path.basename(path), "match": False,
                           "note": "disagreement"})

    match_rate = sum(1 for r in results if r.get("match", False)) / max(len(results), 1)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "n_images": n_compared,
        "top1_agreement": round(match_rate, 4),
        "mean_iou": round(mean_iou, 4),
        "median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
        "pixelwise_results": results,
    }


def compute_speedup(results_list):
    """Compute speedup multiplier of each model vs the first (PyTorch baseline)."""
    if not results_list:
        return {}
    baseline = results_list[0]["avg_ms"]
    speedups = {}
    for r in results_list:
        speedups[r["type"]] = {
            "speedup_vs_pytorch": round(baseline / max(r["avg_ms"], 0.001), 3),
            "avg_ms": r["avg_ms"],
            "avg_fps": r["avg_fps"],
        }
    return speedups


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/face_cnn_best.pth",
                        help="PyTorch model path")
    parser.add_argument("--onnx-fp32", default=None,
                        help="FP32 ONNX model path (auto-detect if not set)")
    parser.add_argument("--onnx-int8", default=None,
                        help="INT8 ONNX model path (skip if not set)")
    parser.add_argument("--images", default="data/face/widerface/WIDER_val/images",
                        help="Directory with test images")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--output", default="reports/logs")
    args = parser.parse_args()

    global cv2
    import cv2

    # Auto-detect ONNX paths if not provided
    model_dir = os.path.dirname(args.model) or "models"
    base_name = os.path.splitext(os.path.basename(args.model))[0]

    onnx_fp32 = args.onnx_fp32 or os.path.join(model_dir, f"{base_name}.onnx")
    onnx_int8 = args.onnx_int8 or os.path.join(model_dir, f"{base_name}_int8.onnx")

    images = get_test_images(args.images, args.max_images)
    if not images:
        print("No test images found. Check --images path.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("FaceCNN Inference Benchmark: PyTorch vs ONNX FP32 vs ONNX INT8")
    print(f"{'='*60}")
    print(f"PyTorch model: {args.model}")
    print(f"ONNX FP32:     {onnx_fp32}")
    print(f"ONNX INT8:     {onnx_int8}")
    print(f"Test images:   {len(images)}")
    print(f"{'='*60}")

    results = []

    # Benchmark PyTorch
    print(f"\n[1/3] Benchmarking PyTorch (TorchScript)...")
    pt_result = benchmark_pytorch(args.model, images)
    results.append(pt_result)
    print(f"  Avg: {pt_result['avg_ms']:.1f}ms, FPS: {pt_result['avg_fps']:.1f}, "
          f"Det rate: {pt_result['detection_rate']:.3f}")

    # Benchmark ONNX FP32
    if os.path.exists(onnx_fp32):
        print(f"\n[2/3] Benchmarking ONNX FP32...")
        onnx_fp32_result = benchmark_onnx_model(onnx_fp32, images)
        results.append(onnx_fp32_result)
        print(f"  Avg: {onnx_fp32_result['avg_ms']:.1f}ms, FPS: {onnx_fp32_result['avg_fps']:.1f}, "
              f"Det rate: {onnx_fp32_result['detection_rate']:.3f}")
    else:
        print(f"\n[2/3] ONNX FP32 not found at {onnx_fp32}, skipping.")
        onnx_fp32_result = None

    # Benchmark ONNX INT8
    if os.path.exists(onnx_int8):
        print(f"\n[3/3] Benchmarking ONNX INT8...")
        onnx_int8_result = benchmark_onnx_model(onnx_int8, images)
        results.append(onnx_int8_result)
        print(f"  Avg: {onnx_int8_result['avg_ms']:.1f}ms, FPS: {onnx_int8_result['avg_fps']:.1f}, "
              f"Det rate: {onnx_int8_result['detection_rate']:.3f}")
    else:
        print(f"\n[3/3] ONNX INT8 not found at {onnx_int8}, skipping.")
        onnx_int8_result = None

    # Compute speedups
    speedups = compute_speedup(results)

    # Detection agreement: PyTorch vs ONNX FP32
    agreement = {}
    if onnx_fp32_result and os.path.exists(onnx_fp32):
        print(f"\n  Computing detection agreement (PyTorch vs ONNX FP32)...")
        agreement_fp32 = compare_detections_pixelwise(
            args.model, onnx_fp32, images, max_compare=min(50, len(images)))
        agreement["pytorch_vs_onnx_fp32"] = agreement_fp32
        print(f"  Top-1 agreement: {agreement_fp32['top1_agreement']:.3f}, "
              f"Mean IoU: {agreement_fp32['mean_iou']:.3f}")

    if onnx_int8_result and os.path.exists(onnx_int8):
        print(f"  Computing detection agreement (PyTorch vs ONNX INT8)...")
        agreement_int8 = compare_detections_pixelwise(
            args.model, onnx_int8, images, max_compare=min(50, len(images)))
        agreement["pytorch_vs_onnx_int8"] = agreement_int8
        print(f"  Top-1 agreement: {agreement_int8['top1_agreement']:.3f}, "
              f"Mean IoU: {agreement_int8['mean_iou']:.3f}")

    # Compile report
    report = {
        "benchmark": {
            "results": results,
            "speedups": speedups,
        },
        "detection_agreement": agreement,
        "config": {
            "pytorch_model": args.model,
            "onnx_fp32": onnx_fp32 if os.path.exists(onnx_fp32) else None,
            "onnx_int8": onnx_int8 if os.path.exists(onnx_int8) else None,
            "n_images": len(images),
        },
    }

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Backend':20s} {'Avg(ms)':>10s} {'FPS':>10s} {'DetRate':>10s} {'Speedup':>10s}"
    print(header)
    print("-" * 60)
    for r in results:
        sp = speedups.get(r["type"], {}).get("speedup_vs_pytorch", 1.0)
        print(f"{r['type']:20s} {r['avg_ms']:>8.1f}ms {r['avg_fps']:>8.1f} "
              f"{r['detection_rate']:>8.3f} {sp:>8.2f}x")
    print("-" * 60)

    # Save report
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "onnx_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
