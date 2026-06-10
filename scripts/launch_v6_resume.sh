#!/bin/bash
# FaceCNN v6.0 Resume — launched $(date)
# Updates:
#   2026-05-26: Disabled mining (--mine-interval 0) to avoid hard-negative
#               mining bug (see reports/HARD_MINING_BUG_REPORT.md).
#               Mining can be re-enabled later with --mine-interval 10.
cd "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.training.train_v6 \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --output models/face_cnn_v6_best.pth \
    --resume models/v6_epochs/v6_epoch_020.pth \
    --epochs 120 --batch-size 16 --lr 1e-3 \
    --warmup-epochs 3 \
    --pos-weight 10.0 --pos-weight-p3 25.0 --pos-weight-p2 50.0 \
    --ema-decay 0.999 \
    --mine-interval 0 \
    --ckpt-interval 5 --diag-interval 10
