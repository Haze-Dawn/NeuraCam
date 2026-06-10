#!/usr/bin/env bash
# =============================================================================
# FaceCNN v7.1 — Phase 1 Launch (Dual-Level: P2+P4, Cosine+SWA)
# =============================================================================
# Fixed: ATSS quality targets (Gaussian center distance), per-level grad clip,
#        bbox bias init -1.0, separate P2/P4 gradient budgets
# Optimized for RTX 2060 (6GB): AMP FP16, batch 14×2=28 eff, 560×768
# Estimated: ~10-13h for 200 epochs
# =============================================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_DIR}"
LOG="${REPO_DIR}/training_phase1_$(date +%Y%m%d_%H%M%S).log"

nohup python3 -u "${REPO_DIR}/src/training/train_v71.py" \
  --data "${REPO_DIR}/data/face/widerface" \
  --output-dir "${REPO_DIR}/models" \
  --batch-size 8 \
  --grad-accum 3 \
  --target-h 560 \
  --target-w 768 \
  --num-workers 4 \
  --checkpointing \
  --lr-backbone 3e-4 \
  --lr-fpn 6e-4 \
  --lr-head 2e-3 \
  --weight-decay 0.05 \
  --varifocal-gamma 2.0 \
  --obj-bias -2.5 \
  --p2-weight 0.5 \
  --swa-start 150 \
  --swa-lr 5e-4 \
  --copy-paste 5 \
  --hard-redistribute 50 \
  --ckpt-interval 5 \
  --ckpt-batch-interval 200 \
  --diag-interval 1 \
  > "${LOG}" 2>&1 &
echo "PID: $!"
echo "Log: ${LOG}"
