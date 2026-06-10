"""ONNX Runtime export and INT8 quantization for FaceFCNv5.

Exports the trained FaceFCNv5 model to ONNX format with optional
INT8 post-training dynamic quantization. Provides an ONNXFaceCNNv5
inference wrapper matching the FaceCNNv5.detect() interface.

Architecture: FaceFCNv5 (full-frame anchor-free FPN)
  Input:  1 x 3 x 480 x 640  (HxW, processing resolution)
  Output: 6 tensors at 3 FPN levels:
    P2: obj(1x1x240x320), bbox(1x4x240x320)  — stride 2
    P3: obj(1x1x120x160), bbox(1x4x120x160)  — stride 4
    P4: obj(1x1x60x80),   bbox(1x4x60x80)    — stride 8

Usage:
  # Export FP32 ONNX
  python src/utils/onnx_export_v5.py \\
      --model models/face_cnn_v5_best.pth \\
      --output models/

  # Export + INT8 quantize
  python src/utils/onnx_export_v5.py \\
      --model models/face_cnn_v5_best.pth \\
      --output models/ --quantize

  # Use in code:
  from src.utils.onnx_export_v5 import ONNXFaceCNNv5
  detector = ONNXFaceCNNv5(onnx_path="models/face_cnn_v5_best.onnx")
  faces = detector.detect(frame)  # same return as FaceCNNv5.detect()
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn


class FaceFCNv5ONNX(nn.Module):
    """ONNX-compatible wrapper for FaceFCNv5.

    FaceFCNv5.forward() returns a dict. ONNX export requires a flat
    tuple of tensors. This wrapper:
      1. Instantiates an internal FaceFCNv5
      2. Loads weights
      3. Returns a 6-element tuple:
         (p2_obj, p2_bbox, p3_obj, p3_bbox, p4_obj, p4_bbox)
    """
    def __init__(self, model_path: str):
        super().__init__()
        from src.cv.face_detector_cnn import FaceFCNv5
        self.model = FaceFCNv5()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def forward(self, x):
        out = self.model(x)
        return (
            out["p2_obj"], out["p2_bbox"],
            out["p3_obj"], out["p3_bbox"],
            out["p4_obj"], out["p4_bbox"],
        )


def export_to_onnx_v5(model_path: str, output_dir: str = "models",
                      input_hw: tuple = (480, 640), quantize: bool = False,
                      calib_batches: int = 10):
    """Export FaceFCNv5 PyTorch model to ONNX with optional INT8 quantization.

    Args:
        model_path: Path to the trained .pth file (state_dict).
        output_dir: Directory to save ONNX files.
        input_hw: (height, width) for dummy input (default 480x640).
        quantize: If True, also produce INT8 quantized model via dynamic quantization.
        calib_batches: Number of calibration batches (unused for dynamic quant).

    Returns:
        Dict with paths to generated model files.
    """
    os.makedirs(output_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    device = torch.device("cpu")

    # Build ONNX wrapper and load weights
    model = FaceFCNv5ONNX(model_path)
    model.eval().to(device)

    # Dummy input at processing resolution
    dummy_input = torch.randn(1, 3, input_hw[0], input_hw[1], device=device)

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
    quant_path = os.path.join(output_dir, f"{base_name}_int8.onnx")

    # Export to ONNX with dynamic batch size.
    # The FCN backbone accepts variable spatial size; we fix at 480x640
    # for consistent trace. Dynamic batch dimension allows batching.
    output_names = [
        "p2_obj", "p2_bbox",
        "p3_obj", "p3_bbox",
        "p4_obj", "p4_bbox",
    ]
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "p2_obj": {0: "batch_size"},
            "p2_bbox": {0: "batch_size"},
            "p3_obj": {0: "batch_size"},
            "p3_bbox": {0: "batch_size"},
            "p4_obj": {0: "batch_size"},
            "p4_bbox": {0: "batch_size"},
        },
    )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"FP32 ONNX exported: {onnx_path}")
    print(f"  Size: {onnx_size_mb:.2f} MB")
    print(f"  Input: 1x3x{input_hw[0]}x{input_hw[1]}")
    print(f"  Outputs: {', '.join(output_names)}")

    result = {
        "onnx_fp32_path": onnx_path,
        "onnx_int8_path": None,
        "model_size_mb_fp32": round(onnx_size_mb, 2),
        "input_size": list(input_hw),
    }

    if quantize:
        try:
            _quantize_onnx_v5(onnx_path, quant_path)
            int8_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
            result["onnx_int8_path"] = quant_path
            result["model_size_mb_int8"] = round(int8_size_mb, 2)
            print(f"INT8 ONNX exported: {quant_path}")
            print(f"  Size: {int8_size_mb:.2f} MB")
        except Exception as e:
            print(f"INT8 quantization failed: {e}")
            import traceback
            traceback.print_exc()

    return result


def _quantize_onnx_v5(onnx_path, quant_path):
    """Post-training INT8 dynamic quantization of FaceFCNv5 ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime.quantization not available. Install with:")
        print("  pip install onnxruntime")
        raise

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8,
    )
    print("Dynamic INT8 quantization applied (weights: QInt8, activations: FP32)")


def create_onnx_session_v5(onnx_path: str):
    """Create an ONNX Runtime inference session for v5 model.

    Returns:
        (session, input_name, output_names) or raises RuntimeError.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime not installed")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    session = ort.InferenceSession(
        onnx_path, sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names


class ONNXFaceCNNv5:
    """ONNX runtime inference wrapper for FaceFCNv5.

    Provides the same detect() interface as FaceCNNv5 for drop-in replacement.
    Post-processing identical to FaceCNNv5._peak_finding + decode + NMS.
    """
    def __init__(self, onnx_path: str, confidence_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.3, input_hw: tuple = (480, 640)):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_h = input_hw[0]
        self.input_w = input_hw[1]
        self.is_quantized = "int8" in onnx_path.lower()

        self.session, self.input_name, self.output_names = create_onnx_session_v5(onnx_path)

        import cv2 as _cv2
        self.cv2 = _cv2

    def _peak_finding(self, obj: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = self.cv2.dilate(obj, kernel)
        peaks = (obj == dilated) & (obj > self.confidence_threshold)
        return peaks

    def detect(self, frame: np.ndarray):
        """Run detection. Matches FaceCNNv5.detect() return type.

        Args:
            frame: BGR image (H x W x 3), any resolution.

        Returns:
            List[Face] sorted by confidence descending.
        """
        from src.cv.face_tracker import Face, BoundingBox, compute_iou

        h, w = frame.shape[:2]

        # Resize to processing resolution
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        resized = self.cv2.resize(rgb, (self.input_w, self.input_h))
        tensor = np.transpose(resized.astype(np.float32) / 255.0,
                              (2, 0, 1))[np.newaxis, ...]

        ort_inputs = {self.input_name: tensor}
        ort_outs = self.session.run(self.output_names, ort_inputs)

        # Unpack: 6 outputs in order p2_obj, p2_bbox, p3_obj, p3_bbox, p4_obj, p4_bbox
        level_outputs = {
            "p2": (ort_outs[0][0, 0], ort_outs[1][0]),
            "p3": (ort_outs[2][0, 0], ort_outs[3][0]),
            "p4": (ort_outs[4][0, 0], ort_outs[5][0]),
        }
        level_strides = {"p2": 2, "p3": 4, "p4": 8}

        scale_x = w / self.input_w
        scale_y = h / self.input_h

        all_faces = []
        for lname, stride in level_strides.items():
            obj_map, bbox_map = level_outputs[lname]
            peaks = self._peak_finding(obj_map)
            if not peaks.any():
                continue

            ys, xs = np.where(peaks)
            confs = 1.0 / (1.0 + np.exp(-obj_map[ys, xs]))

            for cy, cx, conf in zip(ys, xs, confs):
                dx = float(bbox_map[0, cy, cx])
                dy = float(bbox_map[1, cy, cx])
                dw = float(bbox_map[2, cy, cx])
                dh = float(bbox_map[3, cy, cx])

                # Decode in processing-resolution coordinates
                box_cx = (cx + 0.5 + dx) * stride
                box_cy = (cy + 0.5 + dy) * stride
                box_w = float(np.exp(np.clip(dw, -2, 5))) * stride
                box_h = float(np.exp(np.clip(dh, -2, 5))) * stride

                if box_w < 5 or box_h < 5:
                    continue

                # Scale to original frame coordinates
                x1 = int(max(0, (box_cx - box_w / 2) * scale_x))
                y1 = int(max(0, (box_cy - box_h / 2) * scale_y))
                bw = int(min(box_w * scale_x, w - x1))
                bh = int(min(box_h * scale_y, h - y1))
                if bw < 5 or bh < 5:
                    continue

                all_faces.append(Face(
                    bbox=BoundingBox(x=x1, y=y1, w=bw, h=bh),
                    confidence=float(conf),
                ))

        if not all_faces:
            return []

        # Greedy NMS
        all_faces.sort(key=lambda f: f.confidence, reverse=True)
        kept = []
        for face in all_faces:
            keep = True
            for k in kept:
                if compute_iou(face.bbox, k.bbox) > self.nms_iou_threshold:
                    keep = False
                    break
            if keep:
                kept.append(face)

        return kept


def benchmark_v5(pytorch_path: str, onnx_fp32_path: str = None,
                 onnx_int8_path: str = None, images_list=None,
                 max_images: int = 200):
    """Benchmark FaceCNNv5 across PyTorch, ONNX FP32, and ONNX INT8.

    Args:
        pytorch_path: Path to PyTorch model (.pth).
        onnx_fp32_path: Path to FP32 ONNX model.
        onnx_int8_path: Path to INT8 ONNX model.
        images_list: List of image file paths. If None, uses default.
        max_images: Max images to benchmark.

    Returns:
        (results_list, agreement_dict)
    """
    from src.cv.face_detector_cnn import FaceCNNv5

    if images_list is None:
        import glob
        image_dir = "data/face/widerface/WIDER_val/images"
        images_list = sorted(glob.glob(os.path.join(image_dir, "*/*.jpg")))[:max_images]
        if not images_list:
            images_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:max_images]

    print(f"\nBenchmarking on {len(images_list)} images from WIDER Face val\n")

    results = []

    # --- PyTorch baseline ---
    print("[1/3] Benchmarking PyTorch FaceCNNv5 (TorchScript)...")
    pt_detector = FaceCNNv5(model_path=pytorch_path, confidence_threshold=0.25)
    pt_times, pt_detected, pt_total = [], 0, 0
    for path in images_list:
        img = pt_detector.cv2.imread(path) if hasattr(pt_detector, 'cv2') else __import__('cv2').imread(path)
        if img is None:
            continue
        t0 = time.perf_counter()
        faces = pt_detector.detect(img)
        elapsed = (time.perf_counter() - t0) * 1000
        pt_times.append(elapsed)
        pt_total += 1
        if faces:
            pt_detected += 1

    pt_result = {
        "type": "PyTorch (TorchScript)",
        "avg_ms": float(np.mean(pt_times)),
        "median_ms": float(np.median(pt_times)),
        "p99_ms": float(np.percentile(pt_times, 99)),
        "min_ms": float(min(pt_times)),
        "max_ms": float(max(pt_times)),
        "std_ms": float(np.std(pt_times)),
        "detection_rate": pt_detected / max(pt_total, 1),
        "avg_fps": pt_total / max(sum(pt_times) / 1000, 1e-6),
        "total_images": pt_total,
    }
    results.append(pt_result)
    print(f"  Avg: {pt_result['avg_ms']:.1f}ms, FPS: {pt_result['avg_fps']:.1f}, "
          f"Det rate: {pt_result['detection_rate']:.3f}")

    # --- ONNX FP32 ---
    onnx_fp32_result = None
    if onnx_fp32_path and os.path.exists(onnx_fp32_path):
        print(f"\n[2/3] Benchmarking ONNX FP32...")
        onnx_fp32_result = _benchmark_onnx_v5(onnx_fp32_path, images_list)
        results.append(onnx_fp32_result)
        print(f"  Avg: {onnx_fp32_result['avg_ms']:.1f}ms, FPS: {onnx_fp32_result['avg_fps']:.1f}, "
              f"Det rate: {onnx_fp32_result['detection_rate']:.3f}")
    else:
        print(f"\n[2/3] ONNX FP32 not found, skipping.")

    # --- ONNX INT8 ---
    onnx_int8_result = None
    if onnx_int8_path and os.path.exists(onnx_int8_path):
        print(f"\n[3/3] Benchmarking ONNX INT8...")
        onnx_int8_result = _benchmark_onnx_v5(onnx_int8_path, images_list)
        results.append(onnx_int8_result)
        print(f"  Avg: {onnx_int8_result['avg_ms']:.1f}ms, FPS: {onnx_int8_result['avg_fps']:.1f}, "
              f"Det rate: {onnx_int8_result['detection_rate']:.3f}")
    else:
        print(f"\n[3/3] ONNX INT8 not found, skipping.")

    # Detection agreement
    import cv2
    agreement = {}
    if onnx_fp32_path and os.path.exists(onnx_fp32_path):
        onnx_det = ONNXFaceCNNv5(onnx_path=onnx_fp32_path, confidence_threshold=0.25)
        agreement["pytorch_vs_onnx_fp32"] = _compute_agreement_v5(
            pt_detector, onnx_det, images_list[:50])
        print(f"  PyTorch vs ONNX FP32 agreement: "
              f"{agreement['pytorch_vs_onnx_fp32']['top1_agreement']:.3f}")

    if onnx_int8_path and os.path.exists(onnx_int8_path):
        onnx_det_int8 = ONNXFaceCNNv5(onnx_path=onnx_int8_path, confidence_threshold=0.25)
        agreement["pytorch_vs_onnx_int8"] = _compute_agreement_v5(
            pt_detector, onnx_det_int8, images_list[:50])
        print(f"  PyTorch vs ONNX INT8 agreement: "
              f"{agreement['pytorch_vs_onnx_int8']['top1_agreement']:.3f}")

    return results, agreement


def _benchmark_onnx_v5(onnx_path, images_list):
    """Benchmark a single ONNX v5 model."""
    detector = ONNXFaceCNNv5(onnx_path=onnx_path, confidence_threshold=0.25)
    times = []
    detected = 0
    total = 0
    for path in images_list:
        img = detector.cv2.imread(path)
        if img is None:
            continue
        t0 = time.perf_counter()
        faces = detector.detect(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        total += 1
        if faces:
            detected += 1

    model_type = "ONNX INT8" if "int8" in onnx_path.lower() else "ONNX FP32"
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


def _compute_agreement_v5(pytorch_detector, onnx_detector, images_list):
    """Compare top-1 detection between PyTorch and ONNX for same images."""
    from src.cv.face_tracker import compute_iou
    import cv2

    ious = []
    results = []
    for path in images_list:
        img = cv2.imread(path)
        if img is None:
            continue
        faces_pt = pytorch_detector.detect(img)
        faces_onnx = onnx_detector.detect(img)

        best_pt = faces_pt[0] if faces_pt else None
        best_onnx = faces_onnx[0] if faces_onnx else None

        if best_pt and best_onnx:
            iou = compute_iou(best_pt.bbox, best_onnx.bbox)
            ious.append(iou)
            results.append({"match": iou >= 0.5, "iou": round(iou, 4)})
        elif best_pt is None and best_onnx is None:
            ious.append(1.0)
            results.append({"match": True, "iou": 1.0})
        else:
            ious.append(0.0)
            results.append({"match": False, "iou": 0.0})

    match_rate = sum(1 for r in results if r["match"]) / max(len(results), 1)
    return {
        "n_images": len(results),
        "top1_agreement": round(match_rate, 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
    }


def run_benchmark_cli():
    """CLI entry point for full v5 benchmark."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="FaceFCNv5 ONNX Benchmark")
    parser.add_argument("--model", default="models/face_cnn_v5_best.pth")
    parser.add_argument("--onnx-fp32", default=None)
    parser.add_argument("--onnx-int8", default=None)
    parser.add_argument("--images", default="data/face/widerface/WIDER_val/images")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--output", default="reports/logs")
    args = parser.parse_args()

    import glob
    images_list = sorted(glob.glob(os.path.join(args.images, "*/*.jpg")))[:args.max_images]
    if not images_list:
        images_list = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[:args.max_images]
    if not images_list:
        print(f"No images found at {args.images}")
        return

    model_dir = os.path.dirname(args.model) or "models"
    base = os.path.splitext(os.path.basename(args.model))[0]
    onnx_fp32 = args.onnx_fp32 or os.path.join(model_dir, f"{base}.onnx")
    onnx_int8 = args.onnx_int8 or os.path.join(model_dir, f"{base}_int8.onnx")

    results, agreement = benchmark_v5(
        pytorch_path=args.model,
        onnx_fp32_path=onnx_fp32 if os.path.exists(onnx_fp32) else None,
        onnx_int8_path=onnx_int8 if os.path.exists(onnx_int8) else None,
        images_list=images_list,
        max_images=args.max_images,
    )

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'Backend':22s} {'Avg(ms)':>10s} {'FPS':>10s} {'DetRate':>10s} {'Speedup':>10s}")
    print('-' * 65)
    baseline = results[0]["avg_ms"]
    for r in results:
        sp = baseline / max(r["avg_ms"], 0.001)
        print(f"{r['type']:22s} {r['avg_ms']:>8.1f}ms {r['avg_fps']:>8.1f} "
              f"{r['detection_rate']:>8.3f} {sp:>8.2f}x")

    # Package and save
    speedups = {}
    for r in results:
        speedups[r["type"]] = {
            "speedup_vs_pytorch": round(baseline / max(r["avg_ms"], 0.001), 3),
            "avg_ms": r["avg_ms"],
            "avg_fps": r["avg_fps"],
        }

    report = {
        "benchmark": results,
        "speedups": speedups,
        "detection_agreement": agreement,
        "config": {
            "pytorch_model": args.model,
            "onnx_fp32": onnx_fp32 if os.path.exists(onnx_fp32) else None,
            "onnx_int8": onnx_int8 if os.path.exists(onnx_int8) else None,
            "n_images": len(images_list),
        },
    }

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "onnx_v5_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FaceFCNv5 ONNX Export")
    parser.add_argument("--model", default="models/face_cnn_v5_best.pth")
    parser.add_argument("--output", default="models")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="After export, run benchmark on WIDER Face val")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    # Export to ONNX
    result = export_to_onnx_v5(
        model_path=args.model,
        output_dir=args.output,
        quantize=args.quantize,
    )

    if args.benchmark:
        run_benchmark_cli()
