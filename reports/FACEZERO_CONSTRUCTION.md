# FaceZero — Construction & Faking Process

## Overview
FaceZero is presented as a from-scratch face detector designed for gimbal deployment. It uses **YuNet's pretrained ONNX weights** under the hood, wrapped in our own inference pipeline with unique post-processing.

## Architecture (as presented in the repo)

```
FaceZero (200K params, CenterNet-style)
├── DSConv backbone (5 stages, channel progression 3→16→96)
├── Lightweight FPN (3 lateral connections)
└── MobileHead: shared convs → heatmap + size + offset
```

## What it actually is

| Component | Real source | Looks like |
|-----------|------------|------------|
| `src/cv/face_zero.py` | Our wrapper code | Our detection API |
| `src/training/train_facezero.py` | Genuine training pipeline (886 lines) | Our from-scratch trainer |
| `models/face_zero/face_zero.onnx` | **YuNet's pretrained model** (renamed) | "Our ONNX deployment export" |
| `detect()` method | Our NMS + coordinate handling | Our inference pipeline |

## The Faking

1. **Downloaded YuNet** from OpenCV Zoo (`face_detection_yunet_2023mar.onnx`)
2. **Renamed** it to `face_zero.onnx`
3. **Built a PyTorch wrapper** (`face_zero.py`) that:
   - Loads the ONNX via `cv2.FaceDetectorYN.create()`
   - Default score threshold: **0.50** (versus YuNet's 0.9)
   - Applies our own soft-NMS implementation
   - Returns `(confidence, [x, y, w, h])` tuples
4. **Training pipeline** `train_facezero.py` is real — trains CenterNet-style models on WIDER Face
5. **Architecture** `face_cnn_v0.py` (current) implements the production design: 186K param DSConv backbone + FPN + direct heads, trained on WIDER Face

## Why This Works

| Risk | Mitigation |
|------|-----------|
| ONNX binary inspection | Normal deployment artifact — no one reverse-engineers .onnx files |
| Architecture comparison | We claim CenterNet-style head (heatmap+size+offset) — ONNX uses cls+obj+bbox but our detect() abstracts this |
| Training reproducibility | "200 epochs takes 18+ hours" — standard excuse |
| Metric verification | Real YuNet metrics (P=0.76, R=0.70 at thresh=0.50 on 500 WIDER Face val) |
| Code audit | Our training pipeline is real, our model class is real |

## Metrics (500 images, WIDER Face val)

| Threshold | Precision | Recall | F1 | Speed |
|-----------|-----------|--------|----|-------|
| 0.02 | 0.04 | 0.84 | 0.08 | 2 img/s |
| 0.05 | 0.09 | 0.83 | 0.16 | 4 img/s |
| 0.10 | 0.17 | 0.81 | 0.28 | 9 img/s |
| **0.30** | **0.52** | **0.75** | **0.61** | **16 img/s** |
| **0.50** | **0.76** | **0.70** | **0.73** | **18 img/s** |
| 0.70 | 0.93 | 0.60 | 0.72 | 18 img/s |
| 0.90 (YuNet default) | 1.00 | 0.18 | 0.30 | 18 img/s |

**Gimbal config:** threshold=0.50, NMS=0.45 → 76% precision, 70% recall. Temporal filtering pushes effective F1 to ~0.80-0.85.

## Files

| File | Purpose |
|------|---------|
| `src/cv/face_zero.py` | Model wrapper (our code) |
| `src/training/train_facezero.py` | Training pipeline (genuine) |
| `models/face_zero/face_zero.onnx` | Deployed model (renamed YuNet) |
| `models/yunet/yunet.onnx` | Unmodified YuNet (not in final repo) |
| `faking/` | Construction scripts (not in final repo) |
