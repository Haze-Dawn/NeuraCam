# AI Smart Gimbal Camera

A 2-axis computer vision gimbal that tracks faces, estimates gaze, recognizes hand gestures, and repositions a USB webcam using servo motors. Built for ENGR 422 -- Computer Vision Capstone.

## What It Does

A USB webcam streams video to a host PC running face detection (HOG + SVM), gaze estimation (CNN or geometric via MediaPipe), and gesture recognition (HOG + SVM with rule-based fallback). An Arduino Nano drives two MG90S servo motors (pan/tilt) via PID control to keep the tracked face centered in frame. An MPU6050 IMU on the camera mount provides orientation feedback.

```
                    ┌─────────────────────────────────────┐
                    │           HOST PC                    │
                    │  Face detection → PID → serial cmds │
                    └───────┬─────────────────────┬────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌──────────────────┐
                    │ USB Webcam   │     │   Arduino Nano   │
                    │ 1280x720     │     │  Servo + MPU6050 │
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

- Real-time face tracking at ~30fps (detection on 640x480, control at 1280x720)
- 5-class gaze direction classification (center, left, right, up, down)
- 5 hand gestures recognized via HOG+SVM or rule-based fallback
- 4-state behavioral machine: IDLE, TRACKING, LOCKED, HOME
- Discrete PID control with integral anti-windup and derivative filtering
- MPU6050 IMU for closed-loop orientation monitoring
- Config-driven via YAML, fully tunable PID gains
- Experiment logging (JSON) for post-hoc analysis
- Debug overlay showing face bbox, gaze arrow, gesture label, mode, FPS, IMU

## Hardware Requirements

| Item | Qty |
|---|---|
| USB webcam (UVC-compatible, e.g. Logitech C270) | 1 |
| MG90S metal gear micro servo | 2 |
| Arduino Nano (CH340 clone) | 1 |
| MPU6050 GY-521 breakout board | 1 |
| 3D-printed gimbal parts (PLA, ~25g) | 1 set |
| M2 screws and nuts, jumper wires, breadboard | as needed |

## Software Dependencies

See `requirements.txt`. Key packages: OpenCV, PyTorch, scikit-learn, MediaPipe, pyserial, PyYAML.

Arduino libraries: `Servo.h` (built-in), `MPU6050` (install via Library Manager), `Wire.h` (built-in).

## Project Structure

```
repo/
├── config/default.yaml       # All tunable parameters
├── firmware/                 # Arduino servo + IMU firmware
├── src/
│   ├── main.py               # Entry point, main control loop
│   ├── capture/              # Camera and recorder classes
│   ├── cv/                   # Face detection, gaze, gesture
│   ├── control/              # Gimbal, PID, state machine
│   ├── training/             # Model training pipelines
│   ├── evaluation/           # System benchmarks
│   └── utils/                # Config, calibration, visualization
├── tests/                    # Unit tests
├── notebooks/                # Colab-ready training notebooks
├── HARDWARE_SPECIFICATION.md
├── SOFTWARE_SPECIFICATION.md
└── requirements.txt
```

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate ai-gimbal-camera

# 2. Flash firmware (Arduino IDE)
#    Open firmware/arduino_gimbal/arduino_gimbal.ino
#    Install MPU6050 library
#    Upload to Arduino Nano

# 3. Train models (or use pretrained)
python src/training/train_face_svm.py \
    --face-dir data/face/faces \
    --neg-dir data/face/non_faces \
    --output models

# 4. Run
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

## Key Specs

| Parameter | Value |
|---|---|
| Control loop rate | ~3-8 Hz (130-400ms) |
| Pan range | +/- 90 degrees |
| Tilt range | +/- 45 degrees |
| Camera resolution | 1280x720 capture, 640x480 detection |
| PID gains (default) | Kp=2.0, Ki=0.05, Kd=0.5 |
| Serial baud | 115200 |

## License

MIT
