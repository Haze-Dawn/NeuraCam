# AI Smart Gimbal Camera — NeuraCam

A 2-axis computer vision gimbal that tracks faces, recognizes hand gestures, and repositions a USB webcam using servo motors. Built for ENGR 422 — Computer Vision Capstone.

## What It Does

A USB webcam streams 1280×720 video to a host PC running face detection (custom 4-block FCN v3.2, 140K params, FPN skip connection, dilated block4, trained on WIDER Face), 8-state Kalman filter tracking, and 5-gesture recognition (SVM cascade + rule-based fallback). An Arduino Nano drives two EMAX ES08MAII servo motors (pan/tilt) via PID control with adaptive dead zone. An MPU6050 IMU provides closed-loop orientation feedback. Gestures require a 1-second hold to trigger. On face loss, the gimbal performs a 180° search sweep before returning to center.

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
- **Custom FaceCNN (v3.2)** — 4-block fully-convolutional network (140K params, 160M FLOPs) with FPN-style skip connection (block3→1×1 conv → concat with block4), dilated convolution in block4 (dilation=2, RF 56×56 vs 40×40), trained from scratch on WIDER Face (393K faces). Trained with FocalLoss, Gaussian sigma=1.5, full-frame hard-negative mining, GIoU bbox loss, Model EMA weight averaging, and mixed-precision AMP.
- **Score averaging across pyramid scales (v3.2)** — groups overlapping detections across all 5 scales, averages confidence per group, reducing frame-to-frame jitter.
- **Multi-face safe confidence ratio filter (v3.2)** — suppresses false positives below 30% of max confidence but exempts detections that belong to a different face (IoU < 0.1 with max-confidence face).
- **8-state Kalman filter** — constant-velocity model (cx, cy, w, h, vx, vy, vw, vh) with IoU-based matching, sub-pixel smoothing, occlusion prediction (up to 5 lost frames)
- **Confidence-based scale skipping** — skips smaller pyramid scales when detection confidence ≥ 0.9, achieving ~40 FPS on high-confidence frames
- **Multi-scale pyramid** — 5 scales (1.0 → 0.57), detects faces from 1m to 4.5m
- **TorchScript JIT optimized (v3.2)** — forward pass compiled via `torch.jit.script()`, fusing Conv2d+BN+ReLU into single kernels. ~15% CPU speedup, graceful fallback if compilation fails.

### Gesture Control
- **Pose gestures**: OPEN_PALM (lock), FIST (unlock), THUMBS_UP (home). POINT and PEACE classified but no action.
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
- **6-state behavioral machine**: IDLE → TRACKING → LOCKED ↔ TRACKING, TRACKING_HAND, plus HOME and SEARCH
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
Gimbal Camera/
├── config/default.yaml              # All tunable parameters
├── firmware/arduino_gimbal/         # Arduino servo + IMU firmware
├── CAD/                             # Blender 3D gimbal design files
│   ├── Main.blend                   # Gimbal assembly
│   └── Assists/                     # Component reference models
├── Servo Testing/                   # Standalone servo test sketches
├── src/
│   ├── main.py                      # Entry point, capture thread, control loop
│   ├── capture/camera.py            # Camera class (capture thread + Frame dataclass)
│   ├── capture/recorder.py          # MP4 video recorder
│   ├── cv/face_detector_cnn.py      # FaceCNN (4-block FCN) + multi-scale detection
│   ├── cv/face_tracker.py           # KalmanTracker, BoundingBox, Face dataclasses
│   ├── cv/gesture_classifier.py     # HandDetector + GestureClassifier (SVM + rule)
│   ├── control/gimbal.py            # GimbalController (serial, smooth_move, soft endstops)
│   ├── control/pid.py               # PIDController (adaptive dead zone, anti-windup)
│   ├── control/state_machine.py     # 5-state machine (IDLE/TRACKING/LOCKED/HOME/SEARCH)
│   ├── training/train_face_cnn.py   # FaceCNN training (convergence detector, full metrics)
│   ├── training/train_gesture.py    # Gesture SVM training (GridSearchCV + PCA)
│   ├── training/collect_gesture_data.py  # Interactive gesture data collection
│   ├── training/collect_face_data.py     # Face crop collection for fine-tuning
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

Self-collected, 750+ HOG feature vectors (3 subjects, 50 samples per gesture). 1764-dim HOG per sample → PCA-reduced to 80 components. Collection script captures the hand ROI at 64x64, extracts HOG features, and saves to CSV with label 0-4.

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

### Face CNN

RTX 2060 + Ryzen: ~30s/epoch, ~30 min for 50 epochs. CPU: ~20-30 min total.

```bash
PYTHONPATH="." python src/training/train_face_cnn.py \
    --data data/face/widerface \
    --output models/face_cnn.pth \
    --epochs 50 --batch-size 128 --lr 0.001 \
    --sigma 1.5 --focal-gamma 2.0 --focal-alpha 0.25 --min-f1 0.08 \
    --ema-decay 0.999 --amp --multiscale

# Resume from 32/50 checkpoint
PYTHONPATH="." python src/training/train_face_cnn.py \
    --data data/face/widerface \
    --output models/face_cnn.pth \
    --epochs 50 \
    --resume models/face_cnn_epoch_32.pth
```

### Gesture SVM

Optional — the rule-based fallback works without it.

```bash
# 1. Collect data (hold hand in frame, press 0-4 to label)
python src/training/collect_gesture_data.py --output data/gesture/raw/features.csv

# 2. Train SVM (~5s)
python src/training/train_gesture.py \
    --data data/gesture/raw/features.csv \
    --output models --pca-components 80
```

### Face Fine-Tuning (Optional)

```bash
python src/training/collect_face_data.py
```

### 4. Flash Arduino Firmware

Open `firmware/arduino_gimbal/arduino_gimbal.ino` in Arduino IDE, install MPU6050 library, upload to Nano.

### 5. Run

```bash
python src/main.py --config config/default.yaml
```
```

### Controls (while running)
- `q` — quit
- `h` — home gimbal
- `Space` — toggle tracking lock
- `r` — toggle video recording

### Gesture Controls
| Gesture / Action | Trigger | Action | Available In |
|---|---|---|---|
| Open palm 🖐 | Hold ~1s | Lock tracking | TRACKING, TRACKING_HAND |
| Fist ✊ | Hold ~1s | Resume tracking | LOCKED (returns to prior mode) |
| Thumbs up 👍 | Hold ~1s | Home gimbal | Any mode |
| Single wave 👋 | Wave hand side-to-side | Toggle face/hand tracking | TRACKING, TRACKING_HAND |
| Double wave 👋👋 | Two waves in ~1.5s | Toggle zoom mode | TRACKING_HAND |
| Squeeze ✊→🖐 | Continuous (no hold) | Control zoom (3x→1x) | TRACKING_HAND (zoom active) |

### How Wave Detection Works
The system tracks your hand's horizontal position over ~1.5 seconds. When it detects 2+ lateral direction changes with sufficient amplitude (>8% of frame width), it registers a wave. Two waves within 1.5 seconds is a double-wave. This is motion-based, not pose-based — any hand movement works, no specific gesture needed.

### Squeeze Zoom
When tracking your hand (activated by a single wave), zoom is active by default. Close your hand into a fist to zoom in (up to 3x), open it to zoom out. Double-wave to toggle zoom on/off independently. The digital zoom crops the center of the frame and rescales to full resolution.

## Evaluation & Analysis

```bash
# Generate 14 training plots (PDF)
PYTHONPATH="." python src/evaluation/plot_training.py

# WIDER Face mAP against ground truth
PYTHONPATH="." python src/evaluation/evaluate_face_map.py

# Baseline comparison (Haar + MediaPipe vs FaceCNN)
PYTHONPATH="." python src/evaluation/evaluate_baselines.py

# Ablation study (3 epochs per variant for quick test)
PYTHONPATH="." python src/evaluation/run_ablations.py --quick

# PID tuning sweep
PYTHONPATH="." python src/evaluation/tune_pid.py

# System benchmark (requires webcam)
PYTHONPATH="." python src/evaluation/evaluate_system.py --duration 60
```

## Training Pipelines

| Model | Type | Training Data | Time | Output |
|---|---|---|---|---|
| Face FCN v3.2 | 4-block FCN + FPN skip, 140K params | WIDER Face (32K images, 393K faces) | ~25 min GPU (RTX 2060, AMP) | `models/face_cnn_best.pth` |
| Face FCN (fine-tune) | Custom webcam data | Self-collected (50-500 images) | ~5 min | `models/face_cnn_finetuned.pth` |
| Gesture SVM | HOG+PCA+RBF | Self-collected (750 samples, 3 subjects) | ~5s CPU | `models/gesture_{svm,pca,scaler}.pkl` |
| Gesture fallback | Convexity defects | None | — | Always works, ~85-90% accuracy |

## Key Specs

| Parameter | Value |
|---|---|
| Control loop rate | ~6-20 Hz (50-160ms) |
| Face detection FPS | ~21 FPS (47ms without TorchScript), ~25 FPS (~38ms with TorchScript JIT), ~40 FPS with scale skipping |
| Pan range | ±90 degrees (soft endstops) |
| Tilt range | ±45 degrees (soft endstops) |
| Camera resolution | 1280×720 capture, 640×480 detection |
| Face detection | 4-block FCN v3.2, 140K params, 160M FLOPs, RF 56×56, FPN skip, dilated block4 |
| Face tracking | 8-state Kalman filter (IoU match + velocity) |
| PID dead zone | Adaptive: 2-10% (face-size proportional) |
| Gesture accuracy | SVM: 90-95%, Rule-based: 85% |
| Gesture hold time | ~1 second (5 frames at 5fps) |
| Serial format | Batched P:{pan} T:{tilt}\n at 100Hz |
| Serial baud | 115200 |
| State machine | 5 modes: IDLE, TRACKING, LOCKED, HOME, SEARCH |
| SEARCH sweep | 180° pan over 3 seconds on face loss |
| FLOPs (training) | 160M FLOPs/sample → 6.1 TFLOPs/epoch → 305 TFLOPs total |
| FLOPs (inference) | 9.09 GFLOPs at 640×480 (5 scales, +0.8% from v2.0 FPN convs) |
| Model size | 140,420 params, ~563 KB (state dict) |

## License

MIT
