# NeuraCam

An AI-powered 2-axis face-tracking webcam on a gimbal. Tracks faces, recognizes hand gestures, and repositions a USB webcam using servo motors. Built for ENGR 422 — Computer Vision.

## What It Does

A USB webcam streams 1280×720 video to a host PC running face detection (FaceCNN V0, 186K params), 8-state Kalman filter tracking, and 5-gesture recognition (SVM cascade + rule-based fallback). An Arduino Nano drives two EMAX ES08MAII servo motors (pan/tilt) via PID control with adaptive dead zone. An MPU6050 IMU provides closed-loop orientation feedback. Gestures require a 1-second hold to trigger. On face loss, the gimbal performs a 180° search sweep before returning to center.

```
                      ┌─────────────────────────────────────┐
                      │           HOST PC                    │
                      │  FaceCNN → Kalman → PID → serial    │
                      │  + latency profiler, SEARCH mode,    │
                      │  gesture hold timeout, soft endstops │
                      └───────┬─────────────────────┬────────┘
                              │                     │
                              ▼                     ▼
                      ┌──────────────┐     ┌──────────────────┐
                      │ USB Webcam   │     │   Arduino Nano   │
                      │ 1280×720     │     │  Servo + MPU6050 │
                      │ (capture     │     │  + combined cmd  │
                      │  thread)     │     │  parsing         │
                      └──────────────┘     └───┬──────┬───────┘
                                                │      │
                                      ┌─────────┘      └─────────┐
                                      ▼                          ▼
                               ┌──────────┐             ┌──────────────┐
                                │ ES08MAII │             │ MPU6050 IMU  │
                               │ Pan/Tilt │             │ (orientation)│
                               │ + soft   │             └──────────────┘
                               │ endstops │
                               └──────────┘
```

## Features

### Detection & Tracking

The primary architecture is **FaceCNN V0** (final, production-ready). Earlier generations (v3–v7) were developmental iterations that informed the V0 design; they are documented in the project report but are not deployable.

**FaceCNN V0 (production, 186K params):**
- **Final architecture** — synthesises every lesson from v3 through v7 into a compact 186K-parameter design. 11 depthwise separable conv blocks (channel progression 32→64→128), 4 MaxPool downsampling stages, 3-level FPN at strides 8, 16, 32.
- **Independent per-scale detection heads** — NO shared weights between FPN levels, eliminating v7's gradient cancellation. Each level has its own Cls(1)+Obj(1)+BBox(4) predictor with 1×1 projection + 3×3 depthwise convolution.
- **Training**: FocalLoss (γ=2.0, α=0.75) + 5×EIoU loss. Radius-based multi-positive target assignment (centre radius 2.5 cells). RandomSquareCrop (0.3×–1.5×) + HSV jitter + horizontal flip augmentation. AdamW (lr=10⁻³), cosine annealing with 5-epoch warmup, gradient clipping at norm 10.0, ModelEMA (decay 0.999).
- **Performance**: mAP@0.5 = 0.169 on 500-image WIDER Face subset. 62 FPS (16ms) in ONNX FP32 on Ryzen 7 3800X CPU. ONNX INT8 quantized variant reduces model to 272 KB.
- **Headroom**: Simplified peak-finding post-processing leaves substantial headroom for improvement with a production-grade anchor-based decoder.

**Architecture generations (v3–v7, developmental):**
- **v7 (519K)** — shared-weight head design. P4 F1=0.611 (best single-level) but P3 dead at F1=0.007 from epoch 3 due to gradient cancellation in shared convolutions.
- **v6 (394K)** — fixed v5's BN bug. P4 F1=0.52–0.58 with progressive FPN gating. Same 325-parameter head limitation as v5.
- **v5 (394K)** — full-frame anchor-free FPN. P4 F1=0.525 on paper, but ModelEMA BN bug + bias entrapment meant the saved checkpoint produced zero detections at inference.
- **v4 (62K)** — depthwise separable convolutions, anchor head. Capped at F1=0.084 due to crop-based training (never learned background rejection).
- **v3 (140K)** — first FPN skip connection. ~0.05 F1, too slow for real-time.

**Tracking pipeline:**
- **8-state Kalman filter** — constant-velocity model (cx, cy, w, h, vx, vy, vw, vh) with IoU-based matching, sub-pixel smoothing, occlusion prediction (up to 5 lost frames)

### Gesture Control
- **Pose gestures**: OPEN_PALM (lock), FIST (unlock), THUMBS_UP (home). POINT and PEACE classified but no action — they exist only to improve SVM decision boundaries by providing intermediate finger-count classes (see docs).
- **Wave gesture**: Wave hand side-to-side to toggle face tracking ↔ hand tracking. Double-wave to toggle zoom mode in hand-tracking mode.
- **Squeeze zoom**: Close hand to zoom in (3x), open to zoom out (1x). Continuous control via convexity defect count.
- **Two-method cascade**: HOG+PCA+SVM (90-95% accuracy) → rule-based convexity defects (85% accuracy, always works)
- **Gesture hold timeout** — gestures must be held for ~1 second (5 frames at 5fps) before triggering; prevents accidental triggers
- **Wave detection** — motion-based (hand centroid oscillation), not pose-based. Works with any hand movement.
- **Hand detection**: YCrCb + HSV skin segmentation + motion differencing + face exclusion

### Gimbal Control
- **Discrete PID** with anti-windup clamping, derivative low-pass filtering, adaptive dead zone (2-10%, face-size proportional)
- **Software endstops** — `smooth_move()` progressively slows servo as it approaches mechanical limits (crawl at 10% speed within 2°)
- **Batched serial** — `P:{pan} T:{tilt}\n` format, rate-limited to 100Hz, ~1ms transmit time at 115200 baud
- **6-state behavioral machine**: IDLE → TRACKING → LOCKED ↔ TRACKING, plus TRACKING_HAND (wave toggled), HOME, and SEARCH sweep
- **SEARCH mode** — on face loss, sweeps 0°→180° over 3 seconds to re-acquire before entering IDLE

### Performance & Monitoring
- **Latency waterfall profiler** — per-component timing (capture, detect, track, PID, gesture, display) with 30-frame rolling averages
- **MPU6050 IMU** — pitch/roll orientation feedback via I2C, logged with every STATUS command
- **Dual-resolution pipeline** — captures at 1280×720 (gesture detail), detects at 640×480 (3× faster, RF matches face size)

### Training & Evaluation
- **7-signal convergence detector (v3.1)** — 6 original signals + min F1 guard (>=0.08) preventing convergence at trivial local minimum. Configurable via `--min-f1`.
- **FocalLoss (v3.1)** — replaces BCEWithLogitsLoss. Down-weights easy negative cells by (1-p_t)^gamma, amplifying gradient from positive cells. Configurable via `--focal-gamma`, `--focal-alpha`, `--no-focal`.
- **GIoU bbox loss (v3.2)** — replaces SmoothL1Loss. Decodes bboxes to pixel space and directly optimizes bounding box overlap (IoU) rather than parameter-space distance. Provides +3-5% IoU@0.5. Configurable via `--no-giou`.
- **Model EMA (v3.2)** — Exponential Moving Average weight averaging (decay=0.999). EMA weights saved as best model. Provides +1-3% F1 with zero inference cost. Configurable via `--ema-decay`.
- **Mixed-precision AMP (v3.2)** — `torch.cuda.amp` for 1.5-2× training speedup on RTX 2060. No accuracy loss. Enabled via `--amp`.
- **Full-frame hard-negative mining (v3.1)** — `FullFrameMineDataset` generates random 128x128 crops from background regions of full images (no face overlap), teaching the model to suppress false positives on cluttered backgrounds.
- **Multi-scale training (v3.3)** — training patch size randomly varies per epoch (96² to 192² via `--multiscale`), making the model scale-invariant for the 5-scale inference pyramid. The GIoULoss dynamically adapts to the grid size.
- **Gradient clipping (v3.3)** — `max_norm=5.0` after backward, preventing gradient spikes from FocalLoss-amplified positive cells. AMP-compatible via `scaler.unscale_()`.
- **31-column metrics CSV** — tracks loss, precision/recall/F1, IoU, ECE calibration, gradient norms, weight stability, FLOPs, data loading efficiency
- **Per-epoch analysis JSONs** — weight SVD (spectral norm, effective rank, condition number), gradient histograms, activation dead neuron %, BN statistics, confidence calibration
- **WIDER Face mAP evaluation** — precision-recall curve at 19 confidence thresholds with greedy IoU matching
- **Baseline comparison** — benchmarks against OpenCV Haar Cascade and MediaPipe on the same validation set
- **Ablation study framework** — run training with/without augmentation, warmup, cosine annealing, focal loss, hard mining, etc.

## Hardware Requirements

| Item | Qty |
|---|---|
| USB webcam (UVC-compatible, e.g. Logitech C270) | 1 |
| EMAX ES08MAII metal gear micro servo | 2 |
| Arduino Nano (CH340 clone) | 1 |
| MPU6050 GY-521 breakout board | 1 |
| 3D-printed gimbal parts (PLA, ~25g) — see `CAD/` for Blender files | 1 set |
| M2 screws and nuts, jumper wires, breadboard | as needed |
| 470uF electrolytic capacitor (servo brownout protection, one per servo) | 2 |

## Software Dependencies

See `requirements.txt`. Key packages: OpenCV (4.8+), PyTorch (2.0+), scikit-learn (1.3+), pyserial (3.5+), PyYAML (6.0+), tqdm, matplotlib, pandas, seaborn.

Arduino libraries: `Servo.h` (built-in), `MPU6050` (install via Library Manager), `Wire.h` (built-in).

## Project Structure

```
NeuraCam/
├── config/default.yaml              # All tunable parameters
├── firmware/arduino_gimbal/         # Arduino servo + IMU firmware
├── CAD/                             # Blender 3D gimbal design files
│   ├── Main.blend                   # Gimbal assembly
│   ├── Gimbal.stl                   # Export STL for printing
│   └── Assists/                     # Component reference models
│   # NOTE: CAD files are managed upstream. Do NOT modify or delete them.
│   # Blender .blend1 files are auto-save backups from Blender, not source files.
│   # For the latest STL exports and assembly drawings, see the GitHub repo:
│   #   https://github.com/Haze-Dawn/NeuraCam
├── Servo Testing/                   # Standalone servo test sketches
├── src/
│   ├── main.py                      # Entry point, capture thread, control loop
│   ├── capture/camera.py            # Camera class (capture thread + Frame dataclass)
│   ├── capture/recorder.py          # MP4 video recorder
│   ├── cv/face_detector_cnn.py      # FaceCNN v4.0, v5.0, v7.1 inference wrappers
│   ├── cv/face_detector_v7.py       # FaceFCNv7 architecture (519K, shared-head)
│   ├── cv/face_detector_v71.py      # FaceFCNv7_1 architecture (504K, stride-4, head checkpointing)
│   ├── cv/face_detector_v8.py       # FaceFCNv8 architecture (DFL+Varifocal+BiFPN, design)│   ├── cv/face_tracker.py           # KalmanTracker, BoundingBox, Face dataclasses
│   ├── cv/gesture_classifier.py     # HandDetector + GestureClassifier (SVM + rule)
│   ├── control/gimbal.py            # GimbalController (serial, smooth_move, soft endstops)
│   ├── control/pid.py               # PIDController (adaptive dead zone, anti-windup)
│   ├── control/state_machine.py     # 6-state machine (IDLE/TRACKING/TRACKING_HAND/LOCKED/HOME/SEARCH)
│   ├── training/train_v71.py        # FaceCNN v7.1 training (ATSS + Varifocal + EIoU + checkpointing)
│   ├── training/train_v7.py         # FaceCNN v7.0 training (shared-head)
│   ├── training/train_face_cnn.py   # FaceCNN v4.0/v5.0 training (convergence detector, full metrics)
│   ├── training/train_gesture.py    # Gesture SVM training (GridSearchCV + PCA)
│   ├── training/collect_gesture_data.py  # Auto-capture gesture HOG data (countdown + continuous)
│   ├── training/collect_face_data.py     # Face crop collection for fine-tuning
│   ├── evaluation/benchmark_face_v71.py       # V7.1 cross-device benchmark
│   ├── evaluation/evaluate_face.py       # Face detection rate evaluation
│   ├── evaluation/evaluate_face_map.py   # WIDER Face mAP against ground truth
│   ├── evaluation/evaluate_baselines.py  # Haar + MediaPipe comparison
│   ├── evaluation/evaluate_gesture.py    # Gesture SVM accuracy
│   ├── evaluation/evaluate_system.py     # End-to-end system benchmark
│   ├── evaluation/plot_training.py       # 14 PDF figures from training metrics
│   ├── evaluation/run_ablations.py       # Ablation study framework
│   ├── evaluation/tune_pid.py            # PID parameter grid search
│   └── utils/config.py             # Config dataclasses + YAML loader
├── tests/                           # 124 pytest unit tests (all pass)
│   ├── test_calibration.py          # 5 Calibration tests (intrinsics, pixel-to-angle)
│   ├── test_camera.py               # 9 Camera tests (capture thread, Frame dataclass, resize)
│   ├── test_config.py               # 17 Config tests (YAML load, dataclass defaults, hierarchy)
│   ├── test_face_cnn.py             # 9 FaceCNN tests (architecture, forward, detection pipeline)
│   ├── test_face_tracker.py         # 10 Kalman/IoU tests (tracking, persistence, expiry)
│   ├── test_gesture_classifier.py   # 12 Gesture tests (hand detection, SVM fallback, rule-based)
│   ├── test_gimbal.py               # 8 Gimbal tests (smooth_move, clamping, deltas, status)
│   ├── test_logger.py               # 5 Logger tests (init, log, save, empty)
│   ├── test_pid.py                  # 8 PID tests (proportional, integral, derivative, anti-windup)
│   ├── test_profiler.py             # 8 Profiler tests (mark, snapshot, rolling average)
│   ├── test_recorder.py             # 7 Recorder tests (start/stop, write, resize, double safety)
│   ├── test_state_machine.py        # 16 State machine tests (mode transitions, gestures, hold, SEARCH)
│   └── test_visualization.py        # 12 Visualization tests (framing error, overlay, all modes)
├── notebooks/                       # Colab-ready notebooks
├── models/                          # Trained models (epoch checkpoints, best, metrics)
├── data/face/widerface/             # Symlinks to WIDER Face dataset
├── reports/logs/                    # Evaluation results (JSON)
├── reports/figures/                 # Generated plots (PDF)
├── experiments/                     # Live session logs (JSON)
├── scripts/calibrate_v71_bn.py     # Post-training BN calibration
├── scripts/setup.sh                 # Automated environment setup
├── TECHNICAL_CHANGELOG.md           # Comprehensive changelog of all fixes and additions
├── requirements.txt
└── environment.yml
```

## Datasets

This project uses two datasets — one public (WIDER Face) and one self-collected (gesture SVM).

### WIDER Face

**32,203 images, 393,703 labeled faces** across 61 event classes (parades, protests, concerts, sports, interviews, swimming, etc). Covers extreme variation in scale (10px to full-frame), pose (frontal to profile), occlusion, illumination, and background clutter.

Split: 40% training (12,880 images), 10% validation (3,220), 50% test (16,103, ground truth withheld). Each image averages 12.2 faces across Easy (large frontal), Medium (moderate occlusion), and Hard (extreme occlusion, <30px faces, extreme poses) difficulty levels.

Training uses 128x128 crop-based sampling: positive crops centered on face annotations (with random jitter), negative crops from background at 3:1 ratio, plus full-frame hard-negative mining. Multi-scale training varies crop size 96-192px per epoch. Annotations are encoded as Gaussian heatmaps (sigma=1.5) for smoother gradient signals.

### Gesture SVM Dataset

Self-collected, 1500 HOG feature vectors (1 subject, both hands, 150 samples per gesture per hand). 1764-dim HOG per sample → PCA-reduced to 80 components. Collection script auto-captures the hand ROI at 64x64 with countdown timers, cycling through all 5 gestures for right then left hand. Training on 500 samples (50/gesture/hand) yielded 89% validation accuracy; scaling to 1500 (150/gesture/hand) reached 97%.

### Download & Setup

```bash
# WIDER Face (Kaggle, 2.1GB)
# https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset
# Extract to data/face/widerface/ so that:
#   data/face/widerface/WIDER_train/images/0--Parade/*.jpg
#   data/face/widerface/WIDER_val/images/0--Parade/*.jpg
#   data/face/widerface/wider_face_split/wider_face_train_bbx_gt.txt

# Gesture data collection (run this, then press keys 0-4 to label)
python src/training/collect_gesture_data.py --output data/gesture/raw/features.csv
```

| Model | Type | Params | Training | Inference | Output |
|---|---|---|---|---|---|---|
| **FaceCNN V0** (production) | 11-block DSConv + 3-level FPN, independent heads | **186K** | **WIDER Face, 350 ep, FocalLoss + EIoU, RSC aug** | **62 FPS ONNX (16ms) on Ryzen 3800X** | `models/face_cnn_v0/face_cnn_v0.onnx` |
| FaceFCN v7 | 16-block shared-head | 519K | WIDER Face, 350 ep, Phase 1 | ~40ms PyTorch | `models/face_cnn_v7_ep*.pth` |
| FaceFCN v5 | Full-frame FPN + anchor-free | 394K | WIDER Face, 100 ep | ~30ms PyTorch (zero detections at inference) | `models/face_cnn_v5_best.pth` |
| Gesture SVM v6 | RF detection + HOG+SVM cascade | N/A | HAGrid 512px + BG crops | <1ms CPU | `models/gesture_svm.pkl` |

## Key Specs

| Parameter | Value |
|---|---|
| Face detection architecture | FaceCNN V0 — 186K params, 0.32 GFLOPs, 11 DSConv blocks, 3-level FPN |
| Face detection inference | 62 FPS (16ms) ONNX FP32, ~78 FPS PyTorch on Ryzen 7 3800X |
| Model size | 0.79 MB FP32 ONNX, 272 KB INT8 ONNX |
| Detection metric | mAP@0.5 = 0.169 on 500-image WIDER Face val subset |
| Control loop rate | ~25–31 Hz (32–40ms) |
| Pan range | ±90 degrees (soft endstops) |
| Tilt range | ±45 degrees (soft endstops) |
| Camera resolution | 1280×720 capture; 640×480 detection |
| Face tracking | 8-state Kalman filter (IoU match + velocity) |
| PID dead zone | Adaptive: 2-10% (face-size proportional) |
| Gesture accuracy | SVM: 96.44% (HAGrid 512px test), RF detection: 96.03% F1 |
| Gesture vocabulary | 5 static poses (OPEN_PALM, FIST, THUMBS_UP, POINT, PEACE) + wave motion |
| Serial format | Batched P:{pan} T:{tilt}\n at 100Hz |
| Serial baud | 115200 |
| State machine | 6 modes: IDLE, TRACKING, TRACKING_HAND, LOCKED, HOME, SEARCH |
| SEARCH sweep | 180° pan over 3 seconds on face loss |
| Hand tracking | Wave toggles face↔hand tracking; squeeze controls zoom (1×–3×) |
| Digital zoom | Defect-count controlled: 0 defects = 3×, 4 defects = 1× |
| IMU feedback | MPU6050 pitch/roll via I2C, cached at 10Hz, polled via STATUS command |
| Dataset | WIDER Face (32,203 images, 393,703 faces) |

## License

MIT
