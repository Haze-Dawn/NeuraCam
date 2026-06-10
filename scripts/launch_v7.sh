#!/bin/bash
# FaceCNN v7.0 — Max Config Training (200 epochs, ~15.4h)
# Launched: $(date)
# Arch: FaceFCNv7 (519K params, shared head, CIoU, quality loss)
# Schedule: SGDR (T_0=67, 3 cycles), 200 epochs
# Logs: Per-epoch JSON diagnostics, per-epoch metrics CSV, checkpoints every 5 epochs
# Est. time: ~15-16 hours on RTX 2060

cd "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.training.train_v7 \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --output models/face_cnn_v7.pth \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-3 \
    --warmup-steps 500 \
    --sgdr-t0 67 \
    --smoothing 0.1 \
    --bbox-weight 5.0 \
    --grad-accum 2 \
    --ema-decay 0.999 \
    --ckpt-interval 5 \
    --diag-interval 1 \
    --validate-interval 1 2>&1 | tee models/face_cnn_v7_train.log
