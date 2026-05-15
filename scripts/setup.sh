#!/bin/bash
set -e

echo "=== AI Gimbal Camera - Setup ==="

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml 2>/dev/null || \
    conda env update -f environment.yml

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-gimbal-camera

# Create directories
mkdir -p data/face/widerface
mkdir -p data/gesture/raw
mkdir -p models
mkdir -p reports/figures reports/logs
mkdir -p experiments
mkdir -p calibration_images

# Check for serial port
echo ""
echo "Available serial ports:"
python -c "import serial.tools.list_ports; [print(p) for p in serial.tools.list_ports.comports()]" 2>/dev/null || \
    echo "  (pyserial not yet installed, will show after conda setup completes)"

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  1. conda activate ai-gimbal-camera"
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
