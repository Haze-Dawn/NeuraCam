#!/bin/bash
# FaceCNN v6.0 — Max-Effort Fine-Tuning from Epoch 56 Baseline (v4)
# Launched: $(date)
#
# ── What's new in v4 ──
# 1. Flat LR (no decay): Constant 5e-4 signal for all 19 epochs.
#    Previous cosine runs gave ~5 epochs of meaningful signal then flatlined.
# 2. HN accumulation (--hn-cache): Persistent pool across mining rounds.
#    Fresh HNs merged into cache, kept sorted, capped at 768. Retrain uses
#    ALL accumulated HNs — diversity grows each round.
# 3. Positive mixing (--mine-mix-pos 4): Interleaves 4 batches of real faces
#    after HN retrain. Prevents catastrophic forgetting — the model
#    remembers what faces look like while learning to reject HNs.
# 4. Validation-set mining (--mine-val): Mines from val split for HNs the
#    model has never seen in training. More diverse than train-set mining.
# 5. 4x more HNs per round (384 from 4000 images), 8x accumulated (768 max).
#
# ── Epoch schedule ──
# 56: Resume from best checkpoint (mean F1=0.319)
# 57-60: Stabilize at new LR (no mining)
# 61: Mine + retrain (HN cache loaded, fresh HNs merged)
# 62-65: Train normally
# 66: Mine + retrain (cache grows, retrain on ALL accumulated HNs)
# 67-70: Train normally
# 71: Mine + retrain (cache at full capacity ~768)
# 72-75: Final training epochs
#
# ── Expected outcome ──
# Conservative: +0.02-0.03 mean F1 (0.319 → 0.34-0.35)
# Optimistic:  +0.04-0.06 mean F1 (0.319 → 0.36-0.38)
cd "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.training.train_v6 \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --output models/v6_posthoc/face_cnn_v6_finetuned_v4.pth \
    --resume models/face_cnn_v6_best.pth \
    --resume-lr-override 5e-4 \
    --flat-lr \
    --epochs 75 --batch-size 16 \
    --warmup-epochs 0 \
    --p3-obj-start 0 \
    --p2-obj-start 0 \
    --pos-weight 10.0 --pos-weight-p3 25.0 --pos-weight-p2 50.0 \
    --ema-decay 0.999 \
    --mine-interval 5 \
    --mine-start 61 \
    --mine-images 4000 \
    --mine-top-k 384 \
    --hn-cache-max 768 \
    --mine-mix-pos 4 \
    --mine-val \
    --mine-retrain \
    --ckpt-interval 5 --diag-interval 1 \
    --validate-interval 1

# Notes:
#   - --flat-lr: LR stays at 5e-4 (bb), 1e-3 (fpn), 2.5e-3 (heads) all 19 epochs
#   - --mine-start 61: 5 epochs to stabilize before first mining round
#   - --mine-images 4000 --mine-top-k 384: larger mining sweep each round
#   - --hn-cache-max 768: accumulate HNs across rounds, retrain on all
#   - --mine-mix-pos 4: 4 batches of real faces after HN retrain
#   - --mine-val: mine from validation set for unseen negatives
#   - --diag-interval 1: diagnostics every epoch for fine-grained tracking
#   - GPU fragmentation fix: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
