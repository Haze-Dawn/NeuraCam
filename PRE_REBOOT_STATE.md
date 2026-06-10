# FaceCNN v7.1 — Pre-Reboot State Summary

**Date:** June 7, 2026

## Current Status
- Training **not running** — CUDA OOM from zombie GPU process (PID 50751, 5528MiB)
- **Action required:** Reboot, then `bash scripts/launch_v71_phase1.sh`
- Launch script has crash recovery — will auto-resume from `face_cnn_v71_last.pth` if it exists

## What's Fixed (in code, ready for next run)
1. ATSS quality targets: Gaussian center distance (0.7-1.0) instead of cell-to-face IoU (~0.003)
2. ATSS optimized: windowed computation, 1.4ms/face (was ~10ms)
3. P2 optimizer: 4× lower LR (5e-4), weight_decay=0.05
4. P2 gradient clip: max_norm=5 (was 10)
5. Per-level gradient clipping: P2 max_norm=5, others max_norm=100
6. Bbox bias init: -1.0 (was -2.0)
7. hard_redistribute crash: single unsqueeze (was double)
8. Mid-epoch checkpointing: every 200 steps
9. Launch script: auto-resume, batch 8×3=24 effective

## What's Still Broken
- P2 head hmF1=0.0 after 22 epochs (run 3)
- P2 bias moving backward (-2.52→-2.77)
- P2 conv1 norm still growing (10→280 by epoch 20)

## Next Steps After Reboot
1. Run `bash scripts/launch_v71_phase1.sh`
2. Monitor first 10 epochs — check if P2 bias climbs toward 0
3. If P2 still dead at epoch 20: check P2 pos_mask, bbox loss, consider expanding head

## Key Files
- Doc: `Source of truth/FACECNN_v7.1_QFL_SWA_REDESIGN.md`
- Arch: `src/cv/face_detector_v71.py`
- Training: `src/training/train_v71.py`
- Launch: `scripts/launch_v71_phase1.sh`
- Logs: `training_phase1_*.log`
- Checkpoints: `models/face_cnn_v71_*.pth`
