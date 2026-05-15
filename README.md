# AI Smart Gimbal Camera — NeuraCam

A 2-axis computer vision gimbal that tracks faces, recognizes hand gestures, and repositions a USB webcam using servo motors. Built for ENGR 422 — Computer Vision Capstone.

## What It Does

A USB webcam streams 1280×720 video to a host PC running face detection (custom 4-block FCN, 99K params, trained on WIDER Face), 8-state Kalman filter tracking, and 5-gesture recognition (SVM cascade + rule-based fallback). An Arduino Nano drives two MG90S servo motors (pan/tilt) via PID control with adaptive dead zone. An MPU6050 IMU provides closed-loop orientation feedback. Gestures require a 1-second hold to trigger. On face loss, the gimbal performs a 180° search sweep before returning to center.

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
                              │ MG90S    │             │ MPU6050 IMU  │
                              │ Pan/Tilt │             │ (orientation)│
                              │ + soft   │             └──────────────┘
                              │ endstops │
                              └──────────┘
```

## Features

### Detection & Tracking
- **Custom FaceCNN** — 4-block fully-convolutional network (99K params, 155M FLOPs), trained from scratch on WIDER Face (393K faces across 61 event classes)
- **8-state Kalman filter** — constant-velocity model (cx, cy, w, h, vx, vy, vw, vh) with IoU-based matching, sub-pixel smoothing, occlusion prediction (up to 5 lost frames)
- **Confidence-based scale skipping** — skips smaller pyramid scales when detection confidence ≥ 0.9, achieving ~40 FPS on high-confidence frames
- **Multi-scale pyramid** — 5 scales (1.0 → 0.57), detects faces from 1m to 4.5m

### Gesture Control
- **5 gestures**: OPEN_PALM (lock), FIST (unlock), THUMBS_UP (home), POINT (logged only), PEACE (logged only)
- **Two-method cascade**: HOG+PCA+SVM (90-95% accuracy) → rule-based convexity defects (85% accuracy, always works)
- **Gesture hold timeout** — gestures must be held for ~1 second (5 frames at 5fps) before triggering; prevents accidental triggers
- **Hand detection**: YCrCb + HSV skin segmentation + motion differencing + face exclusion

### Gimbal Control
- **Discrete PID** with anti-windup clamping, derivative low-pass filtering, adaptive dead zone (2-10%, face-size proportional)
- **Software endstops** — `smooth_move()` progressively slows servo as it approaches mechanical limits (crawl at 10% speed within 2°)
- **Batched serial** — `P:{pan} T:{tilt}\n` format, rate-limited to 100Hz, ~1ms transmit time at 115200 baud
- **5-state behavioral machine**: IDLE → TRACKING → LOCKED ↔ TRACKING, plus HOME and SEARCH
- **SEARCH mode** — on face loss, sweeps 0°→180° over 3 seconds to re-acquire before entering IDLE

### Performance & Monitoring
- **Latency waterfall profiler** — per-component timing (capture, detect, track, PID, gesture, display) with 30-frame rolling averages
- **MPU6050 IMU** — pitch/roll orientation feedback via I2C, logged with every STATUS command
- **Dual-resolution pipeline** — captures at 1280×720 (gesture detail), detects at 640×480 (3× faster, RF matches face size)

### Training & Evaluation
- **6-signal convergence detector** — monitors val loss, F1, gradient norm, weight cosine similarity, L1 movement, loss slope; stops when model truly plateaus
- **31-column metrics CSV** — tracks loss, precision/recall/F1, IoU, ECE calibration, gradient norms, weight stability, FLOPs, data loading efficiency
- **Per-epoch analysis JSONs** — weight SVD (spectral norm, effective rank, condition number), gradient histograms, activation dead neuron %, BN statistics, confidence calibration
- **WIDER Face mAP evaluation** — precision-recall curve at 19 confidence thresholds with greedy IoU matching
- **Baseline comparison** — benchmarks against OpenCV Haar Cascade and MediaPipe on the same validation set
- **Ablation study framework** — run training with/without augmentation, warmup, cosine annealing, hard mining, etc.

## Hardware Requirements

| Item | Qty |
|---|---|
| USB webcam (UVC-compatible, e.g. Logitech C270) | 1 |
| MG90S metal gear micro servo | 2 |
| Arduino Nano (CH340 clone) | 1 |
| MPU6050 GY-521 breakout board | 1 |
| 3D-printed gimbal parts (PLA, ~25g) — see `CAD/` for Blender files | 1 set |
| M2 screws and nuts, jumper wires, breadboard | as needed |
| 470uF electrolytic capacitor (servo brownout protection) | 1 |

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
│   ├── cv/face_tracker.py           # KalmanTracker class
│   ├── cv/gesture_classifier.py     # HandDetector + GestureClassifier (SVM + rule)
│   ├── control/gimbal.py            # GimbalController (serial, smooth_move, soft endstops)
│   ├── control/pid.py               # PIDController (adaptive dead zone, anti-windup)
│   ├── control/state_machine.py     # 5-state machine (IDLE/TRACKING/LOCKED/HOME/SEARCH)
│   ├── control/kalman.py            # Kalman filter, BoundingBox, Face dataclasses
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
├── tests/                           # 26 pytest unit tests (all pass)
│   ├── test_pid.py                  # 8 PID tests
│   ├── test_gimbal.py               # 8 Gimbal tests (with smooth_move)
│   └── test_face_detector.py        # 10 Kalman/IoU tests
├── notebooks/                       # Colab-ready notebooks
├── models/                          # Trained models (epoch checkpoints, best, metrics)
├── data/face/widerface/             # Symlinks to WIDER Face dataset
├── reports/logs/                    # Evaluation results (JSON)
├── reports/figures/                 # Generated plots (PDF)
├── experiments/                     # Live session logs (JSON)
├── TECHNICAL_IMPLEMENTATION.md      # Full technical documentation
├── requirements.txt
└── environment.yml
```

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate ai-gimbal-camera

# 2. Download WIDER Face dataset
#    https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset
#    Extract to data/face/widerface/ so that:
#      data/face/widerface/WIDER_train/images/0--Parade/*.jpg
#      data/face/widerface/WIDER_val/images/0--Parade/*.jpg
#      data/face/widerface/wider_face_split/wider_face_train_bbx_gt.txt

# 3. Train face CNN
#    RTX 2060 + Ryzen: ~24s/epoch, ~20 min for 50 epochs (convergence detector may stop earlier)
PYTHONPATH="." python src/training/train_face_cnn.py \
    --data data/face/widerface \
    --output models/face_cnn.pth \
    --epochs 50 --batch-size 128 --lr 0.001

# 3b. Resume from checkpoint
PYTHONPATH="." python src/training/train_face_cnn.py \
    --data data/face/widerface \
    --output models/face_cnn.pth \
    --epochs 50 \
    --resume models/face_cnn_epoch_32.pth

# 4. Collect face data (optional — fine-tune on your own camera/subject)
python src/training/collect_face_data.py

# 5. Collect gesture data (optional — rule-based fallback works without it)
python src/training/collect_gesture_data.py --output data/gesture/raw/features.csv

# 6. Train gesture SVM (optional)
python src/training/train_gesture.py \
    --data data/gesture/raw/features.csv \
    --output models --pca-components 80

# 7. Flash firmware (Arduino IDE)
#    Open firmware/arduino_gimbal/arduino_gimbal.ino
#    Install MPU6050 library via Library Manager
#    Upload to Arduino Nano

# 8. Run
python src/main.py --config config/default.yaml
```

### Controls (while running)
- `q` — quit
- `h` — home gimbal
- `Space` — toggle tracking lock
- `r` — toggle video recording

### Gesture Controls
| Gesture | Hold Time | Action |
|---|---|---|
| Open palm 🖐 | ~1s (5 frames) | Lock tracking |
| Fist ✊ | ~1s | Resume tracking |
| Thumbs up 👍 | ~1s | Home gimbal |

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
| Face FCN | 4-block FCN, 99K params | WIDER Face (32K images, 393K faces) | ~20 min GPU (RTX 2060) | `models/face_cnn_best.pth` |
| Face FCN (fine-tune) | Custom webcam data | Self-collected (50-500 images) | ~5 min | `models/face_cnn_finetuned.pth` |
| Gesture SVM | HOG+PCA+RBF | Self-collected (750 samples, 3 subjects) | ~5s CPU | `models/gesture_{svm,pca,scaler}.pkl` |
| Gesture fallback | Convexity defects | None | — | Always works, ~85-90% accuracy |

## Key Specs

| Parameter | Value |
|---|---|
| Control loop rate | ~6-20 Hz (50-160ms) |
| Face detection FPS | ~21 FPS (47ms), ~40 FPS with scale skipping |
| Pan range | ±90 degrees (soft endstops) |
| Tilt range | ±45 degrees (soft endstops) |
| Camera resolution | 1280×720 capture, 640×480 detection |
| Face detection | 4-block FCN, 99K params, 155M FLOPs, ~95% frontal |
| Face tracking | 8-state Kalman filter (IoU match + velocity) |
| PID dead zone | Adaptive: 2-10% (face-size proportional) |
| Gesture accuracy | SVM: 90-95%, Rule-based: 85% |
| Gesture hold time | ~1 second (5 frames at 5fps) |
| Serial format | Batched P:{pan} T:{tilt}\n at 100Hz |
| Serial baud | 115200 |
| State machine | 5 modes: IDLE, TRACKING, LOCKED, HOME, SEARCH |
| SEARCH sweep | 180° pan over 3 seconds on face loss |
| FLOPs (training) | 155M FLOPs/sample → 6 TFLOPs/epoch → 300 TFLOPs total |
| FLOPs (inference) | 9.0 GFLOPs at 640×480 (5 scales) |
| Model size | 99,204 params, ~400 KB (state dict) |

## License

MIT
