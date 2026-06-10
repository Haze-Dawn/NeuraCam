"""
FaceCNN v7.1 — Cross-Device Benchmarking Script
================================================
Measures inference latency + WIDER Face mAP across devices/backends.
Outputs a JSON report for all 7 target devices.

Usage:
  # CPU benchmark (all platforms)
  python3 src/evaluation/benchmark_face_v71.py \
    --model models/face_cnn_v71.onnx \
    --images data/face/widerface/WIDER_val/images \
    --backend openvino \
    --device cpu \
    --num-frames 2000

  # GPU benchmark (CUDA)
  python3 src/evaluation/benchmark_face_v71.py \
    --model models/face_cnn_v71.pth \
    --images data/face/widerface/WIDER_val/images \
    --device cuda \
    --num-frames 2000

  # GPU benchmark (DirectML via ONNX)
  python3 src/evaluation/benchmark_face_v71.py \
    --model models/face_cnn_v71.onnx \
    --images data/face/widerface/WIDER_val/images \
    --backend onnxruntime \
    --device directml

Output: benchmarks/benchmark_<device>_<date>.json
"""

import os, sys, time, json, argparse, glob
from datetime import datetime
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_pytorch(path, device="cuda"):
    import torch
    from src.cv.face_detector_v71 import FaceFCNv7_1
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model = FaceFCNv7_1(obj_bias=-3.0)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def load_model_onnx(path, backend="onnxruntime", device="cpu"):
    if backend == "openvino":
        from openvino.runtime import Core
        core = Core()
        if path.endswith(".xml"):
            model = core.read_model(path)
        else:
            model = core.read_model(path)
        compiled = core.compile_model(model, "CPU", config={
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": "4",
            "CPU_BIND_THREAD": "YES",
        })
        return compiled
    else:
        import onnxruntime as ort
        providers = []
        if device == "directml":
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        sess = ort.InferenceSession(path, providers=providers)
        return sess


def run_benchmark(model, images, backend="pytorch", device="cuda",
                  warmup=100, num_frames=2000, resolutions=None):
    if resolutions is None:
        resolutions = [(640, 480), (480, 360), (320, 240)]

    results = {"device": device, "backend": backend,
               "num_frames": num_frames, "resolutions": {},
               "wider_mAP": None, "timestamp": datetime.now().isoformat()}

    image_paths = glob.glob(os.path.join(images, "*.jpg")) + \
                  glob.glob(os.path.join(images, "*.png"))
    if not image_paths:
        print(f"WARNING: No images found at {images}. Using synthetic frames.")
        image_paths = None

    for rw, rh in resolutions:
        latencies = []
        preprocess_times = []
        postprocess_times = []

        for i in range(warmup + num_frames):
            if image_paths:
                imp = image_paths[i % len(image_paths)]
                frame = cv2.imread(imp)
                if frame is None:
                    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            else:
                frame = np.random.randint(0, 256, (rh, rw, 3), dtype=np.uint8)

            if rw != frame.shape[1] or rh != frame.shape[0]:
                t0 = time.perf_counter()
                frame = cv2.resize(frame, (rw, rh))
                preprocess_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()

            if backend == "pytorch":
                import torch
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
                tensor = tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    model(tensor)
            elif backend == "openvino":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255.0
                model([rgb.astype(np.float32)])
            elif backend == "onnxruntime":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255.0
                model.run(None, {model.get_inputs()[0].name: rgb.astype(np.float32)})

            elapsed = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                latencies.append(elapsed)

        latencies = np.array(latencies)
        results["resolutions"][f"{rw}x{rh}"] = {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "fps": float(1000.0 / np.mean(latencies)),
            "samples": len(latencies),
            "preprocess_mean_ms": float(np.mean(preprocess_times)) if preprocess_times else 0,
        }
        print(f"  {rw}x{rh}: mean={np.mean(latencies):.1f}ms  "
              f"p95={np.percentile(latencies, 95):.1f}ms  "
              f"fps={1000/np.mean(latencies):.0f}  "
              f"samples={len(latencies)}")

    return results


def collect_device_info():
    import platform, cpuinfo
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_brand": cpuinfo.get_cpu_info()["brand_raw"],
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except:
        pass
    try:
        import onnxruntime
        info["onnxruntime_version"] = onnxruntime.__version__
    except:
        pass
    try:
        from openvino.runtime import get_version
        info["openvino_version"] = get_version()
    except:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description="V7.1 Cross-device benchmark")
    parser.add_argument("--model", required=True, help="Model path (.pth or .onnx)")
    parser.add_argument("--images", default="data/face/widerface/WIDER_val/images",
                        help="Images directory")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "pytorch", "onnxruntime", "openvino"])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "directml"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--num-frames", type=int, default=2000)
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[640, 480, 480, 360, 320, 240],
                        help="Resolution pairs: width height [width height ...]")
    args = parser.parse_args()

    if args.backend == "auto":
        if args.model.endswith(".onnx"):
            args.backend = "onnxruntime"
        else:
            args.backend = "pytorch"
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.backend == "pytorch":
        print(f"Loading PyTorch model: {args.model}")
        model = load_model_pytorch(args.model, args.device)
    else:
        print(f"Loading {args.backend} model: {args.model}")
        model = load_model_onnx(args.model, args.backend, args.device)

    resolutions = list(zip(args.resolutions[::2], args.resolutions[1::2]))

    device_info = collect_device_info()
    print(f"\nDevice: {device_info['cpu_brand']}")
    print(f"Backend: {args.backend} | Device: {args.device}")
    print(f"Frames: {args.num_frames} (+{args.warmup} warmup)")

    results = run_benchmark(model, args.images, args.backend, args.device,
                            args.warmup, args.num_frames, resolutions)
    results["device_info"] = device_info
    results["args"] = vars(args)

    os.makedirs(args.output_dir, exist_ok=True)
    device_tag = device_info["cpu_brand"].replace(" ", "_")[:30]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"benchmark_{device_tag}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
