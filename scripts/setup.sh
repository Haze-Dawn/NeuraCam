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
mkdir -p data/raw data/processed data/splits
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
echo "  2. python src/main.py"
echo ""
echo "For gesture data collection:"
echo "  python src/training/collect_gesture_data.py"
echo ""
echo "For model training:"
echo "  python src/training/train_gesture.py"
echo "  python src/training/train_gaze.py --data data/gaze/mpiigaze"
echo ""
echo "For evaluation:"
echo "  python src/evaluation/tune_pid.py"
echo "  python src/evaluation/evaluate_system.py"
