#!/usr/bin/env bash
# =============================================================================
# FaceCNN v8 — Phase 1 Launch (3-Level: P3+P4+P5, BiFPN+DFL, Cosine+SWA)
# =============================================================================
# Optimized for RTX 2060 (6GB): AMP FP16, batch 6×4=24 effective, 480×640
# Estimated: ~18-22h for 250 epochs + 2 pseudo-label cycles
# =============================================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH="${REPO_DIR}"

# Kill nautilus if running — it holds ~22 MiB GPU memory causing OOM
NAUTILUS_PIDS=$(pgrep -x nautilus 2>/dev/null || true)
if [ -n "$NAUTILUS_PIDS" ]; then
    echo "Killing nautilus (holds ~22 MiB GPU RAM)..."
    echo "$NAUTILUS_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 1
fi
LOG="${REPO_DIR}/training_v8_$(date +%Y%m%d_%H%M%S).log"

# Auto-resume: check for existing checkpoint
RESUME_ARGS=()
if [ -f "${REPO_DIR}/models/face_cnn_v8_last.pth" ]; then
    echo "Found existing checkpoint, resuming..."
    RESUME_ARGS=(--resume "${REPO_DIR}/models/face_cnn_v8_last.pth")
fi

nohup python3 -u "${REPO_DIR}/src/training/train_v8.py" \
  --data "${REPO_DIR}/data/face/widerface" \
  --output-dir "${REPO_DIR}/models" \
  --batch-size 4 \
  --grad-accum 4 \
  --target-h 480 \
  --target-w 640 \
  --num-workers 4 \
  --lr-backbone 1e-3 \
  --lr-fpn 2e-3 \
  --lr-head 5e-3 \
  --weight-decay 0.05 \
  --varifocal-gamma 2.0 \
  --p3-weight 1.0 \
  --p4-weight 1.0 \
  --p5-weight 1.0 \
  --swa-start 200 \
  --swa-lr 5e-4 \
  --copy-paste 5 \
  --hard-redistribute 50 \
  --ckpt-interval 5 \
  --ckpt-batch-interval 200 \
  --val-interval 5 \
  --diag-interval 1 \
  --pseudo-cycles 2 \
  --pseudo-epochs 25 \
  "${RESUME_ARGS[@]}" \
  > "${LOG}" 2>&1 &
echo "PID: $!"
echo "Log: ${LOG}"
echo "Monitor: tail -f ${LOG}"
