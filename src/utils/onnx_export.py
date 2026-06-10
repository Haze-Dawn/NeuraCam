"""ONNX Runtime export and INT8 quantization for FaceCNN.

Exports the trained FaceFCN model to ONNX format with optional
INT8 post-training quantization. Provides inference wrappers that
match the FaceCNN.detect() interface for drop-in comparison.

Usage:
  # Export FP32 ONNX model
  python src/utils/onnx_export.py --model models/face_cnn_best.pth --output models/

  # Export + INT8 quantize
  python src/utils/onnx_export.py --model models/face_cnn_best.pth --output models/ --quantize
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn


def export_to_onnx(model_path: str, output_dir: str = "models",
                   input_size: int = 128, num_anchors: int = 3,
                   quantize: bool = False, calib_batches: int = 10):
    """Export FaceFCN PyTorch model to ONNX with optional INT8 quantization.

    Args:
        model_path: Path to the trained .pth file (state_dict).
        output_dir: Directory to save ONNX files.
        input_size: Model input spatial size (default 128).
        num_anchors: Number of anchor boxes (default 3).
        quantize: If True, also produce INT8 quantized model.
        calib_batches: Number of calibration batches for INT8.

    Returns:
        Dict with paths to generated model files.
    """
    os.makedirs(output_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    from src.cv.face_detector_cnn import FaceFCN, FaceCNN

    device = torch.device("cpu")
    model = FaceFCN(num_anchors=num_anchors, use_depthwise=True, use_se=True)
    state_dict = torch.load(model_path, map_location=device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    model.eval().to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
    quant_path = os.path.join(output_dir, f"{base_name}_int8.onnx")

    # Export to ONNX with dynamic batch size.
    # The model is fully-convolutional so it accepts variable input sizes.
    # We keep input size fixed for simplicity during quantization.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"FP32 ONNX model exported to: {onnx_path}")
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Size: {onnx_size_mb:.2f} MB")

    result = {
        "onnx_fp32_path": onnx_path,
        "onnx_int8_path": None,
        "model_size_mb_fp32": round(onnx_size_mb, 2),
    }

    if quantize:
        try:
            _quantize_onnx(model, onnx_path, quant_path,
                          input_size, num_anchors, calib_batches)
            int8_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
            result["onnx_int8_path"] = quant_path
            result["model_size_mb_int8"] = round(int8_size_mb, 2)
            print(f"INT8 ONNX model exported to: {quant_path}")
            print(f"  Size: {int8_size_mb:.2f} MB")
        except Exception as e:
            print(f"INT8 quantization failed: {e}")
            import traceback
            traceback.print_exc()

    return result


def _quantize_onnx(model, onnx_path, quant_path, input_size, num_anchors,
                   calib_batches=10):
    """Post-training INT8 quantization using onnxruntime quantization tools."""
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime not installed. Install with:")
        print("  pip install onnx onnxruntime")
        raise

    # Use dynamic quantization (weights only, activations remain FP32).
    # This is simpler than QDQ (Quantize-Dequantize) static quantization
    # which requires calibration data and is more brittle for variable-size input.
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Dynamic INT8 quantization applied (weights: QInt8, activations: FP32)")


def create_onnx_inference_session(onnx_path: str):
    """Create an ONNX Runtime inference session.

    Args:
        onnx_path: Path to .onnx model file.

    Returns:
        InferenceSession or None if onnxruntime not available.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed.")
        return None

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    session = ort.InferenceSession(
        onnx_path, sess_options,
        providers=["CPUExecutionProvider"],
    )
    return session


class ONNXFaceCNN:
    """Drop-in replacement for FaceCNN that uses ONNX Runtime inference.

    Provides the same detect() interface as FaceCNN for direct comparison.
    """
    def __init__(self, onnx_path: str, confidence_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.25, input_size: int = 128,
                 skip_scale_threshold: float = 0.9,
                 num_anchors: int = 3):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = input_size
        self.skip_scale_threshold = skip_scale_threshold
        self.scale_factor = 1.15
        self.stride = 8
        self.grid_cells = input_size // self.stride
        self.num_anchors = num_anchors
        self.anchor_scales = [1.5, 3.0, 6.0][:num_anchors]
        self.is_quantized = "int8" in onnx_path

        self.session = create_onnx_inference_session(onnx_path)
        if self.session is None:
            raise RuntimeError(f"Failed to create ONNX session for {onnx_path}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        import cv2
        self.cv2 = cv2

    def detect(self, frame: np.ndarray):
        """Matches FaceCNN.detect() interface. Returns list[Face]."""
        from src.cv.face_tracker import Face, BoundingBox, compute_iou

        h, w = frame.shape[:2]
        min_size = self.input_size
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

        n_scales = 5
        scales = [1.0 / (self.scale_factor ** i) for i in range(n_scales)]
        best_conf = 0.0
        all_detections = []

        for si, scale in enumerate(scales):
            if best_conf >= self.skip_scale_threshold:
                continue
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < min_size or new_h < min_size:
                continue

            scaled = self.cv2.resize(rgb, (new_w, new_h))
            pad_h = (self.stride - new_h % self.stride) % self.stride
            pad_w = (self.stride - new_w % self.stride) % self.stride
            if pad_h > 0 or pad_w > 0:
                scaled = self.cv2.copyMakeBorder(
                    scaled, 0, pad_h, 0, pad_w, self.cv2.BORDER_REFLECT)

            tensor = np.transpose(scaled.astype(np.float32) / 255.0,
                                  (2, 0, 1))[np.newaxis, ...]

            ort_inputs = {self.input_name: tensor}
            ort_outs = self.session.run([self.output_name], ort_inputs)
            out = ort_outs[0][0]

            objness = 1.0 / (1.0 + np.exp(-out[0]))
            cells_y, cells_x = np.where(objness > self.confidence_threshold)

            raw_h, raw_w = scaled.shape[:2]
            actual_grid_h = raw_h // self.stride
            actual_grid_w = raw_w // self.stride

            for cy, cx in zip(cells_y, cells_x):
                if cy >= actual_grid_h or cx >= actual_grid_w:
                    continue
                conf = float(objness[cy, cx])

                for ai in range(self.num_anchors):
                    offset = 1 + ai * 3
                    dx = float(out[offset, cy, cx]) if offset < out.shape[0] else 0.0
                    dy = float(out[offset + 1, cy, cx]) if offset + 1 < out.shape[0] else 0.0
                    log_delta = float(out[offset + 2, cy, cx]) if offset + 2 < out.shape[0] else 0.0

                    anchor_size = self.anchor_scales[ai] * self.stride
                    center_x = (cx + 0.5 + dx) * self.stride / scale
                    center_y = (cy + 0.5 + dy) * self.stride / scale
                    box_size = anchor_size * np.exp(log_delta) / scale
                    box_size = max(box_size, 10.0)

                    x1 = int(center_x - box_size / 2)
                    y1 = int(center_y - box_size / 2)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    bw = min(int(box_size), w - x1)
                    bh = min(int(box_size), h - y1)

                    if bw > 5 and bh > 5 and 20 <= box_size <= 500:
                        if conf > best_conf:
                            best_conf = conf
                        all_detections.append(Face(
                            bbox=BoundingBox(x=x1, y=y1, w=bw, h=bh),
                            confidence=conf
                        ))

        if not all_detections:
            return []

        all_detections = sorted(all_detections, key=lambda d: d.confidence, reverse=True)
        groups = []
        assigned = [False] * len(all_detections)
        for i, det in enumerate(all_detections):
            if assigned[i]:
                continue
            group = [det]
            assigned[i] = True
            for j in range(i + 1, len(all_detections)):
                if assigned[j]:
                    continue
                iou = compute_iou(det.bbox, all_detections[j].bbox)
                if iou > self.nms_iou_threshold:
                    group.append(all_detections[j])
                    assigned[j] = True
            groups.append(group)

        merged = []
        for group in groups:
            avg_conf = float(np.mean([d.confidence for d in group]))
            best = max(group, key=lambda d: d.confidence)
            merged.append(Face(
                bbox=BoundingBox(x=best.bbox.x, y=best.bbox.y,
                                 w=best.bbox.w, h=best.bbox.h),
                confidence=avg_conf
            ))

        merged.sort(key=lambda d: d.confidence, reverse=True)
        return self._filter_by_confidence_ratio(merged)

    def _filter_by_confidence_ratio(self, detections):
        from src.cv.face_tracker import compute_iou
        if not detections:
            return []
        max_conf = max(d.confidence for d in detections)
        if max_conf >= 0.5:
            ratio_threshold = 0.3 * max_conf
            kept = []
            for d in detections:
                if d.confidence >= ratio_threshold:
                    kept.append(d)
                else:
                    is_different_face = True
                    for k in kept:
                        if compute_iou(d.bbox, k.bbox) >= 0.1:
                            is_different_face = False
                            break
                    if is_different_face:
                        kept.append(d)
            return kept
        return detections


def benchmark_onnx(onnx_path: str, images, num_warmup=10):
    """Benchmark ONNX model on a list of image paths.

    Returns dict with avg/median/p99 inference time in ms.
    """
    detector = ONNXFaceCNN(onnx_path=onnx_path)
    times = []
    detected = 0
    total = 0

    for path in images:
        img = detector.cv2.imread(path)
        if img is None:
            continue
        import time
        t0 = time.perf_counter()
        faces = detector.detect(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        total += 1
        if faces:
            detected += 1

    times = sorted(times)
    return {
        "avg_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "std_ms": float(np.std(times)),
        "detection_rate": detected / max(total, 1),
        "total_images": total,
        "is_quantized": "int8" in onnx_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/face_cnn_best.pth")
    parser.add_argument("--output", default="models")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--calib-batches", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--num-anchors", type=int, default=3)
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_dir=args.output,
        input_size=args.input_size,
        num_anchors=args.num_anchors,
        quantize=args.quantize,
        calib_batches=args.calib_batches,
    )
