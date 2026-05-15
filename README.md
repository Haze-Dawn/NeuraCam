# AI Smart Gimbal Camera

A 2-axis computer vision gimbal that tracks faces, recognizes hand gestures, and repositions a USB webcam using servo motors. Built for ENGR 422 -- Computer Vision Capstone.

## What It Does

A USB webcam streams video to a host PC running face detection (custom FCN trained on WIDER Face), Kalman filter tracking, and gesture recognition (HOG+PCA+SVM with rule-based fallback). An Arduino Nano drives two MG90S servo motors (pan/tilt) via PID control with adaptive dead zone to keep the tracked face centered in frame. An MPU6050 IMU on the camera mount provides orientation feedback. Serial commands are batched and rate-limited to 100Hz.

```
                    ┌─────────────────────────────────────┐
                    │           HOST PC                    │
                    │  FaceCNN → Kalman → PID → serial    │
                    └───────┬─────────────────────┬────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌──────────────────┐
                    │ USB Webcam   │     │   Arduino Nano   │
                    │ 1280x720     │     │  Servo + MPU6050 │
                    │ (capture     │     │  + combined cmd  │
                    │  thread)     │     │  parsing         │
                    └──────────────┘     └───┬──────┬───────┘
                                              │      │
                                    ┌─────────┘      └─────────┐
                                    ▼                          ▼
                             ┌──────────┐             ┌──────────────┐
                             │ MG90S    │             │ MPU6050 IMU  │
                             │ Pan/Tilt │             │ (orientation)│
                             └──────────┘             └──────────────┘
```

## Features

- Real-time face tracking at ~30fps using custom FCN (3-block, ~24K params, trained on WIDER Face)
- 8-state Kalman filter tracking for smooth trajectories and occlusion prediction
- Adaptive PID dead zone proportional to face distance
- 5 hand gestures via HOG+PCA+SVM or rule-based fallback (convexity defects)
- 4-state behavioral machine: IDLE, TRACKING, LOCKED, HOME
- Dual-resolution pipeline: 1280x720 capture, 640x480 face detection
- Capture thread decouples USB read latency from control loop
- Batched serial commands (P:{pan} T:{tilt}\n) at 100Hz
- MPU6050 IMU for closed-loop orientation monitoring
- Config-driven via YAML, fully tunable
- Experiment logging (JSON) for post-hoc analysis
- Debug overlay showing face bbox, gesture label, mode, FPS, IMU, Kalman uncertainty

## Hardware Requirements

| Item | Qty |
|---|---|
| USB webcam (UVC-compatible, e.g. Logitech C270) | 1 |
| MG90S metal gear micro servo | 2 |
| Arduino Nano (CH340 clone) | 1 |
| MPU6050 GY-521 breakout board | 1 |
| 3D-printed gimbal parts (PLA, ~25g) | 1 set |
| M2 screws and nuts, jumper wires, breadboard | as needed |
| 470uF electrolytic capacitor (servo brownout protection) | 1 |

## Software Dependencies

See `requirements.txt`. Key packages: OpenCV, PyTorch, scikit-learn, pyserial, PyYAML.
No pretrained model dependencies -- all models trained from scratch.

Arduino libraries: `Servo.h` (built-in), `MPU6050` (install via Library Manager), `Wire.h` (built-in).

## Project Structure

```
repo/
├── config/default.yaml       # All tunable parameters (v2.0)
├── firmware/                 # Arduino servo + IMU firmware
│   └── arduino_gimbal/       # Combined command format + MPU6050
├── src/
│   ├── main.py               # Entry point, capture thread, control loop
│   ├── capture/              # Camera (capture thread) and recorder
│   ├── cv/                   # FaceCNN, Kalman tracker, gesture classifier
│   ├── control/              # Gimbal (batched), PID (adaptive dzone), state machine
│   ├── training/             # Face FCN, gesture SVM + PCA training
│   ├── evaluation/           # System, face, gesture benchmarks
│   └── utils/                # Config, calibration, visualization
├── tests/                    # Unit tests
├── notebooks/                # Colab-ready training notebooks
├── HARDWARE_SPECIFICATION.md
├── SOFTWARE_SPECIFICATION.md
├── TECHNICAL_IMPLEMENTATION.md
└── requirements.txt
```

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate ai-gimbal-camera

# 2. Download WIDER Face dataset
#    https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset
#    Extract to data/face/widerface/

# 3. Train face CNN (~2h CPU / ~20min GPU)
python src/training/train_face_cnn.py \
    --data data/face/widerface \
    --output models/face_cnn.pth \
    --epochs 50 --batch-size 64

# 4. Collect gesture data (optional -- rule-based fallback works without it)
python src/training/collect_gesture_data.py --output data/gesture/raw/features.csv

# 5. Train gesture SVM (optional)
python src/training/train_gesture.py \
    --data data/gesture/raw/features.csv \
    --output models --pca-components 80

# 6. Flash firmware (Arduino IDE)
#    Open firmware/arduino_gimbal/arduino_gimbal.ino
#    Install MPU6050 library via Library Manager
#    Upload to Arduino Nano

# 7. Run
python src/main.py --config config/default.yaml
```

### Controls (while running)
- `q` -- quit
- `h` -- home gimbal
- `Space` -- toggle tracking lock
- `r` -- toggle video recording

## Movement Test

To test servos without the full CV pipeline:

```bash
python -c "
from src.control.gimbal import GimbalController
import time
g = GimbalController()
g.home()
time.sleep(1)
for angle in range(45, 135, 5):
    g.set_pan(angle)
    time.sleep(0.02)
for angle in range(45, 135, 5):
    g.set_tilt(angle)
    time.sleep(0.02)
"
```

## Training Pipelines

| Model | Type | Training Data | Time | Output |
|---|---|---|---|---|
| Face FCN | 3-block FCN, 24K params | WIDER Face (32K images, 393K faces) | ~2h CPU / ~20min GPU | `models/face_cnn.pth` |
| Gesture SVM | HOG+PCA+RBF | Self-collected (750 samples, 3 subjects) | ~5s CPU | `models/gesture_{svm,pca,scaler}.pkl` |
| Gesture fallback | Convexity defects | None | -- | Always works, ~85-90% accuracy |

## Key Specs

| Parameter | Value |
|---|---|
| Control loop rate | ~6-20 Hz (50-160ms) |
| Pan range | +/- 90 degrees |
| Tilt range | +/- 45 degrees |
| Camera resolution | 1280x720 capture, 640x480 detection |
| Face detection | FCN, 5-10ms inference, 92-96% frontal |
| Face tracking | 8-state Kalman filter |
| PID dead zone | Adaptive: 2-10% (face-size proportional) |
| Serial format | Batched P:{pan} T:{tilt}\n at 100Hz |
| Hand detection | YCrCb + HSV + motion + face exclusion |
| Serial baud | 115200 |

## License

MIT
