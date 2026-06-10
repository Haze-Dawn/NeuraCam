# FaceCNN v6.0 — Training Results Report

**Date:** May 26-27, 2026
**Status:** Complete (120/120 epochs, exit code 0)
**Hardware:** NVIDIA GeForce RTX 2060 (6 GB VRAM), AMD Ryzen 5 5600X, 8 GB RAM

---

## 1. Executive Summary

FaceCNN v6.0 completed training successfully after recovering from a power loss
interruption at epoch 20. The v5.0 dead-head bug (bias frozen at -4.6, zero
detections) has been definitively fixed. v6.0 sacrifices 0.07 P4 F1 on paper
compared to v5.0's best epoch, but produces actual detections: 2,641 cells
exceed the confidence threshold at epoch 120, with P4 max sigmoid reaching
0.993. Small-face detection (P3/P2) improved 63% and 115% over v5.0 due to
progressive training.

## 2. Training Summary

| Parameter | Value |
|-----------|-------|
| Epochs completed | 120/120 |
| Exit code | 0 |
| Total time | 5.78 hr (epochs 21-120, resumed from checkpoint) |
| Pre-resume time | ~1.3 hr (epochs 1-20, before power loss) |
| Dataset | WIDER Face (12,880 train, 3,220 val) |
| Input resolution | 640×480 full-frame |
| Batch size | 16 |
| Augmentation | H-flip, rotation ±3°, brightness/contrast/saturation |
| NaN events | 0 (skip guard never triggered) |
| VRAM used | 4051-4405 MB (75% of 6 GB) |
| Checkpoints saved | 20 per-epoch + 1 best |
| Diagnostics saved | 12 per-epoch JSON dumps |

## 3. Loss Trajectory

| Epoch | Train Loss | P4 Obj Loss | P3 Obj Loss | P2 Obj Loss | LR |
|-------|-----------|-------------|-------------|-------------|-----|
| 21 | 1.120 | 0.873 | 0.135 | 0.000 | 4.8e-03 |
| 30 | 1.077 | 0.848 | 0.119 | 0.000 | 4.5e-03 |
| 31 | 1.200 | 0.848 | 0.119 | 0.104 | 4.4e-03 |
| 40 | 1.121 | 0.830 | 0.110 | 0.069 | 4.0e-03 |
| 50 | 1.094 | 0.816 | 0.105 | 0.064 | 3.4e-03 |
| 60 | 1.072 | 0.805 | 0.103 | 0.061 | 2.7e-03 |
| 70 | 1.059 | 0.799 | 0.100 | 0.059 | 2.1e-03 |
| 80 | 1.045 | 0.790 | 0.097 | 0.057 | 1.4e-03 |
| 90 | 1.037 | 0.785 | 0.096 | 0.056 | 8.8e-04 |
| 100 | 1.030 | 0.782 | 0.094 | 0.055 | 4.3e-04 |
| 110 | 1.023 | 0.780 | 0.092 | 0.054 | 1.4e-04 |
| 120 | 1.024 | 0.782 | 0.091 | 0.057 | 1.4e-05 |

P2 obj loss activated at epoch 30, causing the spike from 1.077→1.200 at
epoch 31. Loss then steadily declined from 1.20 to 1.03 as the model
stabilized on all three FPN levels.

## 4. F1 Score Per FPN Level

| Epoch | P4 F1 | P3 F1 | P2 F1 | Mean F1 | Best? |
|-------|-------|-------|-------|---------|-------|
| 21 | 0.410 | 0.220 | 0.000 | 0.210 | |
| 25 | 0.392 | 0.281 | 0.000 | 0.224 | |
| 30 | 0.423 | 0.287 | 0.000 | 0.237 | ★ |
| 34 | 0.409 | 0.268 | 0.196 | 0.291 | ★ |
| 40 | 0.389 | 0.209 | 0.138 | 0.245 | |
| 50 | 0.457 | 0.249 | 0.146 | 0.284 | |
| 54 | 0.421 | 0.287 | 0.167 | 0.292 | ★ |
| 56 | 0.468 | 0.285 | 0.204 | 0.319 | ★ BEST |
| 60 | 0.461 | 0.265 | 0.196 | 0.307 | |
| 65 | 0.485 | 0.256 | 0.146 | 0.296 | |
| 70 | 0.461 | 0.248 | 0.155 | 0.288 | |
| 72 | 0.437 | 0.291 | 0.175 | 0.301 | |
| 80 | 0.444 | 0.265 | 0.177 | 0.295 | |
| 90 | 0.443 | 0.251 | 0.147 | 0.280 | |
| 100 | 0.434 | 0.238 | 0.163 | 0.278 | |
| 110 | 0.427 | 0.258 | 0.151 | 0.279 | |
| 120 | 0.451 | 0.254 | 0.157 | 0.287 | |

## 5. Detection Head Confidence

| Epoch | P4 Bias (sigmoid) | P4 Max Sigmoid | cells >0.3 | cells >0.5 |
|-------|-------------------|----------------|------------|------------|
| 1 | -2.450 (0.076) | 0.760 | 23,984 | 168 |
| 20 | -1.122 (0.246) | 0.493 | 0.25 | 0.0 |
| 30 | -0.980 (0.273) | 0.595 | 0.9 | — |
| 40 | -0.889 (0.291) | 0.158 | 0.0 | — |
| 50 | -0.837 (0.302) | 0.615 | 12.1 | — |
| 60 | -0.778 (0.315) | 0.860 | 24.5 | — |
| 70 | -0.741 (0.323) | 1.000 | 2,583 | — |
| 80 | -0.720 (0.327) | 1.000 | 17,913 | — |
| 90 | -0.708 (0.330) | 1.000 | 7,334 | — |
| 100 | -0.703 (0.331) | 0.999 | 4,905 | — |
| 110 | -0.700 (0.332) | 1.000 | 4,484 | — |
| 120 | -0.700 (0.332) | 0.993 | 2,641 | 1,305 |

The head bias stabilized at -0.700 after epoch 100, indicating equilibrium
between the BCEWithLogitsLoss(pos_weight=10) pull and the background-dominated
training data. The model's sigmoid activation gives a base ~33% per-cell
probability, which weight activations modulate to produce confident detections
on real faces (max sigmoid near 1.0) while suppressing background cells.

## 6. v5 vs v6 Comparison

### Architecture (Identical)
Both models use FaceFCNv5: 394K params, depthwise separable backbone (5 stages),
3-level FPN neck (P2/P3/P4), anchor-free detection heads, peak finding + NMS.

### Training Configuration

| Parameter | v5.0 | v6.0 |
|-----------|------|------|
| Loss function | BalancedFocalLoss (γ=2.0) | BCEWithLogitsLoss(pos_weight) |
| Head weight init | Normal(0, 0.01) | Kaiming Normal |
| Head bias init | -4.6 | -2.5 |
| LR schedule | Cosine + restart at epoch 50 | Cosine (no restart) |
| Warmup | None | 3-epoch linear |
| Progressive training | No | P4→P3(ep15)→P2(ep30) |
| Learning rates | Uniform (1e-3) | Diff: 1×, 2×, 5× |
| Head weight decay | 1e-4 | 0.0 |
| EMA BN sync | Broken (v5 bug) | Fixed |
| Logit clamp | None | [-10, +10] |
| NaN guards | None | nan_to_num + grad skip |
| Epochs | 100 | 120 |

### Performance

| Metric | v5 (ep 81) | v6 (ep 120) | v6 (best) |
|--------|------------|-------------|-----------|
| P4 F1 | 0.525 | 0.451 | 0.485 (ep65) |
| P3 F1 | 0.156 | 0.254 | 0.291 (ep72) |
| P2 F1 | 0.073 | 0.157 | 0.204 (ep56) |
| Mean F1 | ~0.251 | 0.287 | 0.319 (ep56) |
| Loss | ~1.00 | 1.024 | — |
| P4 bias (sigmoid) | -4.6 (0.01) | -0.70 (0.332) | — |
| Detections | ZERO | 2,641 cells | — |
| Functional | No | Yes | — |

### Key Insight

v5 scored higher P4 F1 on paper (0.525 vs 0.451) but was non-functional
because the head bias was dead (-4.6, sigmoid 0.01). The BalancedFocalLoss
mathematically compensated for the dead bias during training evaluation,
producing misleading F1 scores. At inference, the dead head produced zero
activations above any reasonable threshold. v6 compensates with a lower P4
F1 ceiling but an actually working detection head.

## 7. Key Milestones

| Epoch | Event |
|-------|-------|
| 1-14 | P4-only training. Backbone learns medium-large face features. |
| 15 | P3 obj loss activates. P3 head begins learning. |
| 20 | Power loss interruption. Training frozen. |
| 21 | Resumed from checkpoint. Mining disabled. |
| 30 | P2 obj loss activates. Loss spikes +11% then decays. |
| 31-55 | All three levels active. Mean F1 climbs from 0.21 to 0.29. |
| 56 | Best mean F1 checkpoint (0.319) saved. |
| 65 | Best P4 F1 (0.485). |
| 70 | Head reaches full confidence (max sigmoid 1.0). |
| 80 | Detection firing peaks: 17,913 cells >0.3. |
| 80-120 | Fine-tuning phase. LR decays 1.4e-3 → 1.4e-5. |
| 120 | Training complete. Head bias stabilized at -0.700. |

## 8. Head Weight Evolution

| Epoch | head_p4 L2 | head_p3 L2 | head_p2 L2 | fpn_lat2 L2 | fpn_lat3 L2 | fpn_lat4 L2 |
|-------|-----------|-----------|-----------|-------------|-------------|-------------|
| 1 | 1.460 | 1.423 | 1.471 | 11.318 | 16.085 | 22.847 |
| 20 | 1.484 | 1.449 | 1.471 | 11.241 | 16.038 | 22.825 |
| 120 | 1.696 | 2.002 | 1.833 | 10.295 | 15.173 | 22.166 |

P3 head weights grew the most (+38%), consistent with P3 having the most
room for improvement (v5 P3 F1 was only 0.156). FPN lateral weights
compressed slightly (-3 to -8%) as the backbone improved and required
less aggressive projection. All trends are healthy — no weight collapse
or explosion.

## 9. BatchNorm Final State

All BN layers show healthy statistics at epoch 120. No frozen (γ=1.0,
σ=0.0) layers detected. Running means and variances are well-distributed
across stages, with deeper stages showing higher variance accumulation
as expected. The EMA BN sync fix successfully propagated running
statistics to the saved checkpoint — confirmed by the non-uniform BN
γ values (range: 0.91-1.00) and non-zero γ standard deviations
(0.01-0.04 per layer).

## 10. Bugs Encountered and Fixed

1. **First-run failure:** pos_weight=99/300/500 + Kaiming init → loss=43.28,
   NaN gradients at batch 211. Fixed by reducing pos_weights to 10/25/50
   and adding logit clamp + nan_to_num + grad NaN skip.

2. **Hard-negative mining crash:** Unbounded candidate list (1-3 GB CPU RAM)
   + DataLoader num_workers=2 with OpenCV fork deadlock → exponential
   slowdown (13-27 s/it). Fixed by bounded min-heap (96 items) and
   num_workers=0. Mining disabled for training.

3. **Power loss at epoch 20:** Training interrupted during mining pass.
   All state dicts preserved in v6_epoch_020.pth. Resumed successfully.

## 11. Output Files

| File | Description |
|------|-------------|
| `models/face_cnn_v6_best.pth` | Best checkpoint (epoch 56, mean F1=0.319) |
| `models/v6_epochs/v6_epoch_*.pth` | 20 per-epoch checkpoints (every 5 epochs) |
| `models/v6_diagnostics/v6_epoch_*.json` | 12 per-epoch diagnostic dumps |
| `models/face_cnn_v6_best_metrics.csv` | Full per-epoch metrics (100 rows) |
| `models/v6_resume_train.log` | Complete training log (epochs 21-120) |
| `models/archive/mining_bug_investigation/` | Pre-interruption evidence archive |

---

*Document Version: 1.0 — May 27, 2026*
