#!/usr/bin/env bash
# =============================================================================
# FaceCNN v7.1 — Optimized Training v2 (QFL + Cosine/SWA)
# =============================================================================
# Key changes from v1:
#   Loss:        VarifocalLoss + MSE(iou) → QualityFocalLoss (β=2.0)
#               Removes iou head (1 output channel, 1 loss term, all iou diag)
#   Scheduler:   SGDR T_0=65 → Cosine annealing (no restart) + SWA@ep150
#               Cosine avoids the restart degradation proven in v6 (§13.7)
#               SWA final 25% averages weights for better generalization
#   Bias init:  obj_bias=-3.0 → -2.5 (v6-proven value, sigmoid 0.076 vs 0.047)
#   Precision:  AMP FP16 → FP32 with --no-amp (FP16 kills VFL numerically)
#               Use --checkpointing to save VRAM instead (stage-level)
#   Guards:     Logit clamp [-10,10], exp pre-clamp max=5.0, NaN counter
#   Resolution: 560×768 (grid: 140×192 = 26,880 cells, +40% vs 480×640)
#   Batch:      8 (FP32 at 560×768 fits in 6GB with checkpointing)
#               Previously: 24 (FP16) → forced batch 8 with --no-amp anyway
#
# Why QFL beats VFL: VFL at init (σ=0.047): background cells get p^γ=0.002
# weight → positive gradient diluted 560× by mean reduction. QFL at same init:
# |0-0.047|^β=0.002 for background, |1-0.047|^β=0.91 for positives → adaptive
# imbalance handling without pos_weight knobs.
#
# Why Cosine+SWA beats SGDR: v6 proved SGDR restarts cause -8.7% F1 drop and
# need 20 epochs to recover (Final Report §13.7). Pure cosine + SWA gives:
#   - No restart shock (smooth decay through epoch 150)
#   - SWA final 25% at higher LR (5e-4) prevents cosine death zone
#   - SWA weight averaging finds wider minima → better generalization
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_DIR}"

PHASE="${1:-1}"

case "${PHASE}" in
  1)
    exec python3 "${REPO_DIR}/src/training/train_v71.py" \
      --data "${REPO_DIR}/data/face/widerface" \
      --output-dir "${REPO_DIR}/models" \
      --batch-size 8 \
      --grad-accum 1 \
      --num-workers 4 \
      --checkpointing \
      --no-amp \
      --lr-backbone 3e-4 \
      --lr-fpn 6e-4 \
      --lr-head 2e-3 \
      --weight-decay 0.05 \
      --qfl-beta 2.0 \
      --obj-bias -2.5 \
      --swa-start 150 \
      --swa-lr 5e-4 \
      --copy-paste 5 \
      --hard-redistribute 50 \
      --ckpt-interval 5 \
      --diag-interval 1
    ;;
  2)
    echo "=== Phase 2: Cross-dataset fine-tune (50 epochs) ==="
    echo "  Fine-tunes from Phase 1 SWA checkpoint."
    echo "  Low LR (backbone 1.5e-4, head 6e-4) adapts to new distributions"
    echo "  without destroying converged features."
    echo "  Adds MAFA (masked), FDDB (profiles), UFDD (conditions), IJB-C (poses)."
    echo ""
    BEST="${REPO_DIR}/models/face_cnn_v71_swa.pth"
    [ -f "$BEST" ] || { echo "No SWA checkpoint — run Phase 1 first"; exit 1; }
    exec python3 "${REPO_DIR}/src/training/train_v71.py" \
      --data "${REPO_DIR}/data/face/widerface" \
      --datasets mafa,fddb,ufdd,ijbc \
      --output-dir "${REPO_DIR}/models" \
      --batch-size 8 \
      --grad-accum 1 \
      --num-workers 4 \
      --checkpointing \
      --no-amp \
      --lr-backbone 1.5e-4 \
      --lr-fpn 3e-4 \
      --lr-head 6e-4 \
      --resume "${BEST}"
    ;;
  *)
    echo "Usage: $0 {1|2}"
    echo "  Phase 1 — WIDER-only (200 epochs, QFL+Cosine+SWA)"
    echo "  Phase 2 — Cross-dataset fine-tune (50 epochs)"
    exit 1
    ;;
esac
