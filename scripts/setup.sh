#!/bin/bash
set -e

echo "=== AI Gimbal Camera - Setup ==="

# Ensure pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo "pyenv not found. Install it from https://github.com/pyenv/pyenv"
    exit 1
fi

# Create directories
mkdir -p data/face/widerface
mkdir -p data/gesture/raw
mkdir -p models
mkdir -p reports/figures reports/logs
mkdir -p experiments
mkdir -p calibration_images

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check for serial port
echo ""
echo "Available serial ports:"
python -c "import serial.tools.list_ports; [print(p) for p in serial.tools.list_ports.comports()]" 2>/dev/null || \
    echo "  (pyserial not yet installed, run: pip install pyserial)"

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  1. Ensure you're in the right pyenv environment"
echo "  2. Download WIDER Face from: https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset"
echo "     Extract to data/face/widerface/"
echo "  3. Train face CNN:  python src/training/train_face_cnn.py --data data/face/widerface --output models/face_cnn.pth"
echo "  4. python src/main.py"
echo ""
echo "Optional:"
echo "  Gesture data collection:      python src/training/collect_gesture_data.py"
echo "  Gesture SVM training:         python src/training/train_gesture.py"
echo "  PID tuning sweep:             python src/evaluation/tune_pid.py"
echo "  System benchmark:             python src/evaluation/evaluate_system.py"
