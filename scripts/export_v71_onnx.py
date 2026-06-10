"""
FaceCNN v7.1 — ONNX Export + OpenVINO IR + Quantization
========================================================
Exports trained PyTorch model to ONNX (opset 17), then optionally:
  1. Post-training dynamic INT8 quantization (ONNX Runtime fallback)
  2. QAT INT8 model export (Intel VNNI primary path — requires --qat flag)
  3. OpenVINO IR conversion (Intel CPU primary inference backend)

Usage:
  # Standard ONNX export (FP32)
  python3 scripts/export_v71_onnx.py \
    --model models/face_cnn_v71_best.pth \
    --output models/face_cnn_v71.onnx

  # QAT model export (after --qat training)
  python3 scripts/export_v71_onnx.py \
    --model models/face_cnn_v71_qat.pth \
    --output models/face_cnn_v71_qat.onnx \
    --qat

  # Full pipeline: ONNX + INT8 + OpenVINO IR
  python3 scripts/export_v71_onnx.py \
    --model models/face_cnn_v71_best.pth \
    --output models/face_cnn_v71 \
    --int8 --openvino
"""

import os, sys, argparse, warnings
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv.face_detector_v71 import FaceFCNv7_1

warnings.filterwarnings("ignore")


def export_onnx(model, output_path, input_shape=(1, 3, 480, 640),
                opset=17, qat=False):
    model.eval()
    device = next(model.parameters()).device

    dummy = torch.randn(*input_shape, device=device)

    if qat:
        from torch.ao.quantization import fuse_modules_qat
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model, inplace=False)
        model.eval()
        model = torch.ao.quantization.convert(model, inplace=False)

    output_names = ["obj", "iou", "bbox", "p2_obj", "p2_iou", "p2_bbox"]
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "obj": {0: "batch_size"},
            "iou": {0: "batch_size"},
            "bbox": {0: "batch_size"},
            "p2_obj": {0: "batch_size"},
            "p2_iou": {0: "batch_size"},
            "p2_bbox": {0: "batch_size"},
        },
    )
    print(f"ONNX exported: {output_path}")
    print(f"  Input:  {list(input_shape)}")
    print(f"  Outputs: obj(1,1,H,W) iou(1,1,H,W) bbox(1,4,H,W)")
    print(f"           p2_obj(1,1,2H,2W) p2_iou(1,1,2H,2W) p2_bbox(1,4,2H,2W)")
    print(f"  Opset:  {opset}")


def quantize_dynamic_onnx(model_path, output_path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(model_path, output_path, weight_type=QuantType.QInt8)
        print(f"INT8 quantized: {output_path}")
    except ImportError:
        print("WARNING: onnxruntime not installed. Skipping INT8 quantization.")
        print("Install: pip install onnxruntime")


def convert_openvino(model_path, output_dir):
    try:
        import subprocess
        basename = os.path.splitext(os.path.basename(model_path))[0]
        out_xml = os.path.join(output_dir, f"{basename}.xml")
        if not os.path.exists(out_xml):
            subprocess.run([
                "mo", "--input_model", model_path,
                "--output_dir", output_dir,
                "--compress_to_fp16",
            ], check=True)
            print(f"OpenVINO IR: {out_xml}")
        else:
            print(f"OpenVINO IR exists: {out_xml}")
    except FileNotFoundError:
        print("WARNING: OpenVINO Model Optimizer (mo) not found. Skipping IR conversion.")
        print("Install: pip install openvino openvino-dev")


def main():
    parser = argparse.ArgumentParser(description="V7.1 ONNX + OpenVINO export")
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output path (.onnx or prefix)")
    parser.add_argument("--qat", action="store_true", help="QAT model (contains fake-quantize nodes)")
    parser.add_argument("--int8", action="store_true", help="Also export INT8 quantized")
    parser.add_argument("--openvino", action="store_true", help="Also convert to OpenVINO IR")
    parser.add_argument("--input-shape", type=int, nargs=4, default=[1, 3, 480, 640],
                        help="Input shape: batch channels height width")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.qat:
        print("Loading QAT model...")
        ckpt = torch.load(args.model, map_location="cpu")
        sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
        model = FaceFCNv7_1(obj_bias=-2.5)
        model.load_state_dict(sd, strict=True)
        onnx_path = args.output if args.output.endswith(".onnx") else args.output + "_qat.onnx"
    else:
        print(f"Loading model: {args.model}")
        ckpt = torch.load(args.model, map_location="cpu")
        sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
        model = FaceFCNv7_1(obj_bias=-2.5)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            print(f"  Strict load failed — attempting non-strict")
            model.load_state_dict(sd, strict=False)
        onnx_path = args.output if args.output.endswith(".onnx") else args.output + ".onnx"

    export_onnx(model, onnx_path, args.input_shape, args.opset, args.qat)

    if args.int8:
        int8_path = onnx_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic_onnx(onnx_path, int8_path)

    if args.openvino:
        out_dir = os.path.dirname(onnx_path) or "."
        convert_openvino(onnx_path, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
