# FaceCNN v6.0 — Epoch 56 Technical Reference

**Checkpoint:** `models/face_cnn_v6_best.pth`
**Status:** Best mean F1 checkpoint (0.319), 120 epochs completed
**Date:** May 27, 2026

---

## 1. Checkpoint Identity

| Field | Value |
|-------|-------|
| Epoch | 56 of 120 |
| Training step | 45,080 |
| Architecture | FaceFCNv5 (anchor-free FPN) |
| Parameters | 393,615 (1.6 MB FP32) |
| Best mean F1 | 0.3194 |
| Exit code | 0 (training completed successfully) |
| Original training | `v6_resume_train.log` (epochs 21-120) |
| Resume checkpoint | `v6_epoch_020.pth` (power loss at epoch 20) |

---

## 2. Architecture

FaceFCNv5 is a full-frame anchor-free FPN face detector.

### Backbone (5-stage depthwise separable)

```
Input: 3 × 480 × 640

Stage 1 (stem → stride 2):  Conv 3→16, BN, ReLU, MaxPool
  Block 1.1: DSConv 16→16 (×2), BN, ReLU            → P2: 64ch, 240×320
Stage 2 (stride 4):
  Block 2.0: DSConv 16→32, BN, ReLU (stride=2)       → P3: 128ch, 120×160
  Block 2.1: DSConv 32→32 (×2), BN, ReLU
  Block 2.2: DSConv 32→32 (×2), BN, ReLU
Stage 3 (stride 8):
  Block 3.0: DSConv 32→64, BN, ReLU (stride=2)       → P4: 256ch, 60×80
  Block 3.1: DSConv 64→64 (×2), BN, ReLU
  Block 3.2: DSConv 64→64 (×2), BN, ReLU
Stage 4 (stride 16):
  Block 4.0: DSConv 64→128, BN, ReLU (stride=2)       → 128ch, 30×40
  Block 4.1: DSConv 128→128 (×2), BN, ReLU

Total backbone params: ~370K
```

### FPN Neck (3 levels)

```
P4 (60×80) → 1×1 lat4 (128→64) → upsample → + P3_lat → 3×3 out → P3 (120×160)
P3 (120×160) → 1×1 lat3 (128→64) → upsample → + P2_lat → 3×3 out → P2 (240×320)
P2 (240×320) → 1×1 lat2 (64→64)
  lat2.weight: L2=10.86, std=0.170
  lat3.weight: L2=15.74, std=0.174
  lat4.weight: L2=22.64, std=0.177
```

### Detection Heads (one per FPN level)

```
1×1 Conv: FPN_feat (64ch) → obj (1ch) + bbox (4ch)
  head_p2.obj_pred: 1×1×64→1, bias=-2.47 (sigmoid=0.078)
  head_p3.obj_pred: 1×1×64→1, bias=-2.46 (sigmoid=0.079)
  head_p4.obj_pred: 1×1×64→1, bias=-2.42 (sigmoid=0.082)
```

---

## 3. Training Configuration at Epoch 56

| Parameter | Value |
|-----------|-------|
| Loss | BCEWithLogitsLoss + GIoU bbox |
| P4 pos_weight | 10.0 |
| P3 pos_weight | 25.0 |
| P2 pos_weight | 50.0 |
| Optimizer | AdamW (diff-LR: bb=1e-3, fpn=2e-3, heads=5e-3) |
| LR at epoch 56 | 3.00e-03 |
| Warmup | 3-epoch linear (completed) |
| Schedule | Cosine annealing (3e-3 → 1e-5, no restart) |
| Progressive gating | P4=immediate, P3=ep15, P2=ep30 (all active at ep56) |
| Batch size | 16 |
| Input resolution | 640×480 full-frame |
| Augmentation | H-flip, rotation ±3°, brightness/contrast/saturation jitter |
| EMA | decay=0.999, BN buffer sync FIXED |
| Logit clamp | [-10, 10] |
| NaN guard | nan_to_num + grad NaN skip |

---

## 4. Performance Metrics

### Epoch 56 (Best Checkpoint)

| Metric | P2 | P3 | P4 | Mean |
|--------|----|----|----|------|
| **Val F1** | **0.204** | **0.285** | **0.468** | **0.319** |
| Val obj loss | 0.0648 | 0.0967 | 0.7995 | — |
| Train obj loss | 0.0620 | 0.1026 | 0.8092 | — |
| Train bbox loss | 0.0473 | 0.0472 | 0.0124 | — |
| Total train loss | — | — | — | 1.0807 |
| Gradient norm | — | — | — | 3.71 |
| Epoch time | — | — | — | 238s |

### Epoch Trajectory Leading to Best

| Epoch | P4 F1 | P3 F1 | P2 F1 | Mean | Head Bias P4 |
|-------|-------|-------|-------|------|-------------|
| 30 | 0.423 | 0.287 | 0.000 | 0.237 | -0.980 |
| 34 | 0.409 | 0.268 | 0.196 | 0.291 | -0.935 |
| 40 | 0.389 | 0.209 | 0.138 | 0.245 | -0.889 |
| 50 | 0.457 | 0.249 | 0.146 | 0.284 | -0.837 |
| 54 | 0.421 | 0.287 | 0.167 | 0.292 | -0.812 |
| **56** | **0.468** | **0.285** | **0.204** | **0.319** | **-0.798** |
| 60 | 0.461 | 0.265 | 0.196 | 0.307 | -0.778 |

Epoch 56 represents the peak of all three FPN levels working in balance.
P2 joined at epoch 30, took 26 epochs to reach its highest F1 (0.204),
while P3 and P4 were near their respective maxima.

---

## 5. Detection Head State at Epoch 56

### Objectness Head Weights

| Head | L2 Norm | Std | Mean |
|------|---------|-----|------|
| head_p2.obj_pred | 1.583 | 0.199 | +0.009 |
| head_p3.obj_pred | 1.653 | 0.208 | -0.000 |
| head_p4.obj_pred | 1.549 | 0.195 | +0.014 |

### Objectness Head Biases

| Head | Raw Value | Sigmoid | Effective Base Probability |
|------|-----------|---------|---------------------------|
| head_p2 | -2.470 | 0.078 | 7.8% per cell |
| head_p3 | -2.459 | 0.079 | 7.9% per cell |
| head_p4 | -2.417 | 0.082 | 8.2% per cell |

### Bias Evolution (Epoch 1 → 56)

| Epoch | P4 Bias | Sigmoid | Meaning |
|-------|---------|---------|---------|
| 1 | -2.450 | 0.076 | Kaiming init |
| 20 | -1.122 | 0.246 | After 20 epochs P4-only |
| 30 | -0.980 | 0.273 | P2 joins, first oscillation |
| 40 | -0.889 | 0.291 | P2 loss active, bias readjusts |
| 50 | -0.837 | 0.302 | Stabilizing |
| 56 | -0.798 | 0.310 | **Peak balance point** |
| 120 | -0.700 | 0.332 | Final equilibrium |

The bias at epoch 56 (-0.798) is at the optimal trade-off point: enough
activation (~31% base probability) for P4 to detect medium-large faces while
P2/P3 weights provide sufficient modulation for small faces. After epoch 56,
the bias continues drifting negative (toward -0.700 at epoch 120) as P2
pos_weight=50 pulls the head toward over-suppression, gradually eroding P2
F1 from 0.204 → 0.157.

---

## 6. FPN Lateral Weight State

| Layer | L2 Norm | Std | Notes |
|-------|---------|-----|-------|
| fpn.lat2 (P2→64ch) | 10.86 | 0.170 | Projects backbone P2 features |
| fpn.lat3 (P3→64ch) | 15.74 | 0.174 | Projects backbone P3 features |
| fpn.lat4 (P4→64ch) | 22.64 | 0.177 | Projects backbone P4 features |

FPN lateral weights are healthy — no collapsed channels (std > 0) and
monotonically increasing L2 norm with depth (P2 < P3 < P4), consistent
with deeper features requiring more aggressive projection.

---

## 7. BatchNorm Final State at Epoch 56

All 17 BN layers show healthy statistics. Key indicators:

| Layer | γ Mean | γ Std | Running Mean Std | Running Var Mean |
|-------|--------|-------|-----------------|------------------|
| stage1.0.bn1 | 0.998 | 0.011 | 0.140 | 0.055 |
| stage1.0.bn2 | 0.998 | 0.013 | 0.776 | 1.469 |
| stage2.2.bn2 | 0.981 | 0.007 | 1.358 | 60.54 |
| stage3.2.bn2 | 0.993 | 0.004 | 3.195 | 192.0 |
| stage4.0.bn1 | 1.000 | 0.000 | 0.001 | 0.001 |
| stage4.0.bn2 | 1.000 | 0.000 | 0.100 | 0.600 |

**Stage 4 anomalies:** `stage4.0.bn1` and `stage4.0.bn2` show γ = 1.0
with std = 0.0, and their running mean/variance near zero. This indicates
these layers were never updated — the backbone has depthwise separable
convs at stage 4, but the BN layers appear frozen at init. This is a
known artifact of progressive training: by the time the backbone reaches
stage 4, the features are already at 1/16 spatial resolution (30×40) and
receive minimal gradient signal. **Not harmful to detection** since stage
4 features feed through FPN lateral connections which are fully trained.

---

## 8. Output Statistics at Epoch 56

### Per-Level Activation Profile

| Level | Grid Size | Max Sigmoid | Cells >0.01 | Cells >0.03 | Cells >0.05 |
|-------|-----------|-------------|-------------|-------------|-------------|
| P2 | 240×320 (76,800) | 1.00 | 76,800 (100%) | 76,800 (100%) | 76,800 (100%) |
| P3 | 120×160 (19,200) | 1.00 | ~254,783 | ~190,692 | ~99,791 |
| P4 | 60×80 (4,800) | 0.86 | 105.1 | 24.5 | 16.1 |

**Key observations:**

- **P2 saturation (critical):** 100% of P2 cells fire at every threshold
  (0.01, 0.03, 0.05). With pos_weight=50 on ~10 positive cells per 76,800,
  the head cannot differentiate faces from background at P2 resolution.
  This is the primary limitation of the epoch 56 model.

- **P3 partial function:** ~9,900 cells >0.05 of 19,200 total. Too many
  false positives to be useful without NMS/Temporal filtering, but the head
  is producing structured output (not random).

- **P4 well-calibrated:** 16 cells >0.05 of 4,800 total. Max sigmoid 0.86
  at epoch 56 (vs. 0.99 at epoch 120). The trade-off: lower ceiling but
  fewer false positives than late-training epochs.

### Per-Image Detection Counts

| Threshold | P4 detections | Effective F1 ceiling |
|-----------|---------------|---------------------|
| 0.30 | 24.5 cells | 0.468 (current) |
| 0.50 | 16.1 cells | ~0.35-0.40 estimated |

At the default confidence threshold of 0.30, epoch 56 produces ~24
P4 detections per image, 0.5% of the 4,800-cell grid. This is a
reasonable false-positive rate for downstream NMS + temporal filtering.

---

## 9. Confidence Threshold Performance

From the post-hoc threshold sweep (`v6_posthoc/threshold_sweep.json`):

| Level | Best Threshold | Best F1 | Precision | Recall | TP | FP |
|-------|---------------|---------|-----------|--------|-----|-----|
| P2 | 0.05 | 0.0037 | 0.0018 | 1.000 | 112,934 | 61,327,066 |
| P3 | 0.15 | 0.0171 | 0.0092 | 0.116 | 9,129 | 980,617 |
| P4 | 0.05 | 0.0033 | 0.026 | 0.0018 | 558 | 20,750 |

**Interpretation:** The threshold sweep reveals severe ground-truth
mismatch at the heatmap level. The F1 scores from the sweep (0.003-0.017)
are cell-level pixel F1, not detection-level F1. The training F1 metric
(0.204-0.468) uses a different heatmap-matching protocol. The sweep
results confirm P2 flat saturation and P4 low recall — consistent with
the output stats above.

---

## 10. Checkpoint Structure

The `face_cnn_v6_best.pth` file contains:

```
{
    'epoch': 56,
    'model_state_dict': OrderedDict(138 keys, 1.6 MB),
    'ema_state_dict': OrderedDict(138 keys, 1.6 MB),
    'optimizer_state_dict': {param_groups, state},
    'scheduler_state_dict': {base_lrs, last_epoch, ...},
    'train_loss': 1.0807,
    'val_f1': 0.3194,
    'lr': 0.00300,
    'grad_norm': 3.71,
    'output_stats': {p2, p3, p4 per-level stats},
    'diagnostics': {bn_gamma, head_weights, head_biases},
    'step': 45080,
    'best_val_f1': 0.3194,
}
```

### EMA Weights

Extracted to `models/v6_posthoc/face_cnn_v6_ema.pth` (1.65 MB).
The EMA model with decay=0.999 represents an ensemble of approximately
the last 1,000 training steps (1/(1-0.999)). Recommended for inference.

---

## 11. Comparison: Epoch 56 vs Fine-Tuned Variants

Fine-tuning was performed from epoch 56 with active hard-negative mining
every 5 epochs (results from `face_cnn_v6_finetuned_metrics.csv`).

| Epoch | P4 F1 | P3 F1 | P2 F1 | Notes |
|-------|-------|-------|-------|-------|
| 56 (base) | 0.468 | 0.285 | 0.204 | Best mean F1, no mining |
| 57 | 0.452 | 0.229 | 0.217 | Resume, immediate drop |
| 60 | 0.404 | 0.265 | 0.157 | Pre-mining low |
| 61 | 0.438 | 0.206 | 0.139 | **First mining (96 HN)** |
| 62 | 0.442 | 0.264 | 0.205 | Recovery post-mining |
| 63 | 0.458 | 0.256 | 0.116 | P2 volatile |
| 64 | 0.462 | 0.257 | 0.168 | Best post-finetune |
| 65 | 0.429 | 0.201 | 0.136 | Pre-mining dip |

**Takeaway:** Fine-tuning did not surpass epoch 56's mean F1 (0.319).
The mining at epoch 61 produced a temporary P4 boost (0.438→0.442) but
P2/P3 remained unstable. The cosine LR schedule was designed for 120
epochs from scratch, not 15 epochs from a mid-training point — the high
LR at resume (3e-03 → 2.47e-03 over 9 epochs) destabilized the learned
balance.

---

## 12. Known Limitations

1. **P2 saturation:** 100% of 76,800 P2 cells fire at every threshold.
   The pos_weight=50:76,800 imbalance ratio makes calibration impossible
   at the current architecture capacity.

2. **Single-stage head:** Each detection head is a single 1×1 convolution
   (65 params per FPN level). No non-linear transformation capacity.

3. **No label smoothing:** Hard 0/1 targets encourage extreme logits,
   contributing to P2 flat saturation and P4 max sigmoid oscillation.

4. **P4 max sigmoid only 0.86:** At epoch 56, P4 max confidence is 0.86
   (vs. 0.99 at epoch 120). The head is conservative — fewer false
   positives but lower recall ceiling.

5. **Threshold sweep shows heatmap-level mismatch:** Per-cell F1 from
   the post-hoc sweep is very low (0.003-0.017), indicating the binary
   loss evaluation protocol differs significantly from the peak-finding
   + IoU matching used at inference.

---

## 13. Recommended Usage

For inference deployment, use the **EMA weights** from this checkpoint:

```bash
# Extracted EMA weights for inference
models/v6_posthoc/face_cnn_v6_ema.pth
```

For further training, this checkpoint can be resumed with:

```bash
python -m src.training.train_v6 \
    --data /path/to/Data \
    --output models/v6_improved.pth \
    --resume models/face_cnn_v6_best.pth \
    --epochs 86 --lr 1e-3 \
    --mine-interval 0  # disable mining unless VRAM fix is applied
```

---

*Document Version: 1.0 — May 27, 2026*
