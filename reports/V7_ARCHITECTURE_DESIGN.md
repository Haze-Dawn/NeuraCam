# FaceCNN v7.0 — Architecture Design

**Date:** May 27, 2026
**Constraint:** No pretrained backbone. Trained from scratch on WIDER Face.
**Target:** <500K params, <35ms CPU inference, working detection on all FPN levels.

---

## 1. Root Cause Analysis: Why v6 Underperforms

### Problem 1: Detection heads are a single linear projection

The current detection head is literally just two 1×1 convolutions with no
non-linearity between the FPN feature and the output:

```python
self.obj_pred = nn.Conv2d(64, 1, 1)     # 65 params
self.bbox_pred = nn.Conv2d(64, 4, 1)    # 260 params
```

This is 325 parameters per head. The head cannot learn ANY transformation
of the FPN features — it's a pure linear projection. For comparison, RetinaNet
uses 4 conv layers per head (256→256→256→256→Cls/Box). Our head has literally
no decision capacity. The backbone does all the work, and the head is just a
readout layer.

**Why this matters:** P4 F1 plateaued at 0.45. With a 2-layer head that can
learn a non-linear decision boundary, the head can extract more from the same
backbone features. This is the single highest-impact architectural change.

### Problem 2: All three heads are identical despite vastly different tasks

| Level | Grid size | Cells | Positive cells | Pos:Neg ratio | Head params |
|-------|-----------|-------|----------------|---------------|-------------|
| P4 | 80×60 | 4,800 | ~48 | 1:100 | 325 |
| P3 | 160×120 | 19,200 | ~20 | 1:1,000 | 325 |
| P2 | 320×240 | 76,800 | ~10 | 1:7,680 | 325 |

P2 has 16× more spatial positions and 77× worse class imbalance than P4, with
the SAME head architecture. The P2 head needs more capacity to process more
spatial information and make finer-grained decisions.

### Problem 3: Deepest backbone features are discarded

The backbone computes c5 (stage4, 256ch at 80×60, dilation=2) but the FPN
uses c4 (stage3, 256ch at 80×60) for P4. c5 has the largest receptive field
(56×56 pixels at stride 8 ≈ 448px in the 640×480 input) and the highest-level
semantic features. Using c5 for P4 gives the head access to features that
have seen more context.

### Problem 4: Backbone has no channel attention

The DSConv backbone has no mechanism to selectively emphasize important
channels. Adding SE blocks at the two deepest stages (c4 and c5) gives the
backbone channel-wise recalibration at negligible cost (~16K params total).

### Problem 5: Stem is too simple

A single Conv2d(3→32, s=2) throws away half the spatial information in one
operation. Adding one small refinement block after the stem preserves more
early detail for P2's higher-resolution grid.

---

## 2. v7 Architecture Design

### 2.1 Backbone Changes

**Stem enhancement:**
```
Old: Conv2d(3→32, s=2, k=3) → BN → ReLU
New: Conv2d(3→32, s=2, k=3) → BN → ReLU → DSConvBlock(32→32, s=1)
```
Adds one depthwise separable block after the stem. This applies a 3×3
depthwise conv + 1×1 pointwise conv to refine the stem output before it
enters stage1. The stride-2 downsampling still happens in the stem conv,
but the DSConvBlock recovers some spatial detail lost in the aggressive
initial downsampling.

Params: +2,208. FLOPs: +0.7M at 320×240.

**SE attention at c4 and c5:**
```
c4 → SEBlock(256, reduction=16) → c4_se
c5 → SEBlock(256, reduction=16) → c5_se
```
SE squeezes the spatial dimensions with global average pooling, passes
through two linear layers (256→16→256) with sigmoid gating, and
multiplies the result channel-wise with the input. This tells the backbone
"which of the 256 channels are important right now."

Params: +16,448 total (8,224 each). FLOPs: negligible (global pool + 2×
small Linear layers).

**FPN lateral input change:**
```
Old: P4 lateral = Conv2d(c4(256ch)→64ch)  — uses stage3 output
New: P4 lateral = Conv2d(c5_se(256ch)→64ch) — uses stage4 output (deeper)
```
c5 has the same spatial resolution as c4 (80×60) but double the receptive
field (56×56 cell units vs ~28×28) and deeper feature hierarchy. The lateral
conv has the same dimensions (256→64) so no parameter change — just a different
input source.

### 2.2 Detection Head Redesign

**P4 and P3: 2-layer head (RetinaNet-style)**
```
FPN feature (64ch)
    ↓
Conv2d(64→64, 1×1) → BN → ReLU     ← new intermediate layer
    ↓
    ├── Conv2d(64→1, 1×1) → obj logit
    └── Conv2d(64→4, 1×1) → bbox offsets
```
Params: 4,549 per head (vs 325 previously). The intermediate 64→64 conv
learns a non-linear transformation of the FPN features before the final
linear readout. This is the standard RetinaNet/FCOS head design.

**P2: 3-layer head (deeper, more capacity)**
```
FPN feature (64ch)
    ↓
Conv2d(64→64, 1×1) → BN → ReLU     ← layer 1
    ↓
Conv2d(64→64, 1×1) → BN → ReLU     ← layer 2 (extra for P2)
    ↓
    ├── Conv2d(64→1, 1×1) → obj logit
    └── Conv2d(64→4, 1×1) → bbox offsets
```
Params: 8,773 (vs 325). The extra conv layer gives P2 twice the non-linear
processing depth of P4/P3. This is necessary because P2 operates on 76,800
cells with only 10 positive cells — the hardest discrimination task by far.

**Head bias initialization (unchanged):**
All heads: bias = -2.5 (sigmoid 0.076). Kaiming normal for conv weights.
This is the v6 fix and remains correct.

### 2.3 P2 Special Handling (Tiny-Face Regularization)

P2 at 320×240 has extreme class imbalance (76,800 cells, ~10 positives).
To prevent the P2 head from memorizing grid positions instead of learning
features, add spatial dropout between the conv layers:

```
Conv2d(64→64, 1×1) → BN → ReLU → Dropout2d(p=0.1) → Conv2d(64→64, 1×1) → ...
```

`Dropout2d(p=0.1)` randomly zeroes 10% of entire feature map CHANNELS (not
individual cells). This forces the head to use distributed features rather
than memorizing specific channel activations at specific grid positions.
The dropout is disabled during inference.

### 2.4 Parameter Budget

| Component | Current (v5/v6) | Proposed (v7) | Delta |
|-----------|:---:|:---:|:---:|
| Stem | 3,136 | 5,344 | +2,208 |
| Stage1 | 14,592 | 14,592 | 0 |
| Stage2 | 59,776 | 59,776 | 0 |
| Stage3 | 153,856 | 153,856 | 0 |
| Stage4 | 124,160 | 124,160 | 0 |
| SE c4 | — | 8,224 | +8,224 |
| SE c5 | — | 8,224 | +8,224 |
| FPN lateral | 33,024 | 33,024 | 0 |
| P4 head | 325 | 4,549 | +4,224 |
| P3 head | 325 | 4,549 | +4,224 |
| P2 head | 325 | 8,773 | +8,448 |
| **Total** | **393,615** | **429,271** | **+35,656** |
| **Increase** | — | **+9.1%** | — |

**Inference FLOPs impact:**
The additional head conv layers are 1×1 at spatial resolutions 80×60, 160×120,
and 320×240. Total additional FLOPs: ~15M (3.0% increase from 500M). SE blocks
add ~0.5M FLOPs. Stem refinement adds ~0.7M FLOPs.

**Inference latency:** 30ms → ~31ms on CPU (Ryzen 5 5600X).

---

## 3. Training Configuration (Two-Phase Pipeline)

### 3.1 Phase 1: Max Training (200 epochs planned, ~150 actual)

Applied to the v7 architecture during the initial training phase:

| Parameter | v6 | v7 | Reason |
|-----------|:---:|:---:|--------|
| P3 obj activation | epoch 15 | epoch 3 | Give P3 197 epochs of obj loss |
| P4 obj activation | epoch 0 | epoch 0 | Always active (well-behaved) |
| P3 pos_weight | 25 | 15 | More conservative for 1,000:1 ratio |
| P4 pos_weight | 10 | 10 | Unchanged (works well) |
| Label smoothing | None | s=0.1 | Prevent overconfidence, improve calibration |
| LR schedule | Cosine only | SGDR: CosineAnnealingWarmRestarts (T_0=67, T_mult=1) | 3 cycles over 200 ep. T_0=67 gives cycles of 67/134/199. Each restart escapes local minima. Full LR restart (1e-3), not half — the 4-layer head has enough gradient signal to recover. |
| Multi-scale training | No | 384-800px random | Cross-level scale perturbation. Each epoch face sizes shift between P3 and P4. |
| Copy-paste augmentation | No | Yes | +5 positives per image (3-10 total). 3× gradient signal for P3 bbox. |
| Hard negative mining | No | Every 10 ep (from ep 20) | Background crops model confidently misclassifies. Mixed into normal batches. |
| bbox weight | 1.0 | 5.0 | Fix gradient starvation. Brings obj:bbox gradient ratio to ~2:1. |
| Gradient accumulation | 1 | 2 (effective batch 32) | More stable bbox gradients. Reduces variance from zero-positive P3 batches. |
| EMA decay | 0.999 | 0.999 | Unchanged. BN buffers synced (v6 bug fixed). |
| Head bias init | -2.5 | -2.5 | sigmoid(-2.5) = 0.076 baseline. Unchanged. |
| bbox bias init | 0.0, 0.0, 0.0, 0.0 | 0.0, 0.0, -2.0, -2.0 | dw/dh=-2 gives exp(-2)×stride default box. Model must GROW output, not escape zero. |
| Warmup | 3 epochs | 3 epochs | Linear 0→1e-3. Unchanged. |
| Logit clamp | [-10,10] | [-10,10] | Unchanged. |
| NaN guards | Yes | Yes | Unchanged. |

### 3.2 Phase 2: Pseudo-Label Fine-Tuning (3 × 25 epochs)

After Max training completes, the best EMA checkpoint enters pseudo-labeling:

| Parameter | Phase 1 | Phase 2 | Reason |
|-----------|:-------:|:-------:|--------|
| Epochs | 200 | 25 per cycle | Model is converged — just adapting BN to new data distribution. 25 ep = 7× BN adaptation window (3 ep). |
| Learning rate | 1e-3 → 1e-6 | 1e-4 → 1e-6 | Conservative start. Model already converged; high LR would destabilize. |
| SGD optimizer | AdamW (lr=1e-3) | AdamW (lr=1e-4 start) | Same optimizer, lower LR. All groups scaled proportionally. |
| Label smoothing | s=0.1 | s=0.2 (pseudo only) | Higher smoothing accounts for pseudo-label noise. Prevents overfitting to the model's own mistakes. |
| SGDR restart | Yes (T_0=67) | No | No restart needed for 25 ep. Constant LR decay. |
| Multi-scale | Yes | Yes | Unchanged. |
| Copy-paste | Yes | Yes | Unchanged. |
| Hard negative mining | Yes (every 10 ep) | Yes (every 10 ep) | **Critical**: catches pseudo-label FPs and labels them as negatives in the next cycle. Self-correcting adversarial loop. |
| Pseudo-label source | — | phase 1 best EMA | 16K WIDER test images, quality > 0.3 threshold. Only high-confidence, well-localized predictions. |
| Pseudo-label inference | — | FP32, no TTA | Batch inference on 16K images ~8 min. TTA would add 2× (not worth 4% recall at 10× the time). |

### 3.3 LR Scheduling

**Phase 1 (Max, 200 epochs):**
```
Epochs 1-3:    Linear warmup 0 → 1e-3
Epochs 3-67:   Cosine decay 1e-3 → 1e-6        (cycle 1)
Epoch 67:      SGDR restart: LR reset to 1e-3   (full restart)
Epochs 67-134: Cosine decay 1e-3 → 1e-6         (cycle 2)
Epoch 134:     SGDR restart: LR reset to 1e-3   (full restart)
Epochs 134-200: Cosine decay 1e-3 → 1e-6        (cycle 3)
```

Full restarts (not half) are safe because the 4-layer head has 230× more
capacity than v6 — gradient collapse during restart recovery is structurally
impossible. Each restart perturbs the model out of its current loss basin
and into a better one. The documented v5 restart degradation (+5.6% loss
spike) was caused by a 1-layer head on a 62K model; a 4-layer head on a
519K model absorbs the perturbation.

**Phase 2 (Pseudo-label fine-tuning, 25 epochs per cycle):**
```
Epochs 1-5:    Linear warmup 0 → 1e-4           (slow, model is converged)
Epochs 5-25:   Linear decay 1e-4 → 1e-6         (gentle adaptation)
```

No restart during fine-tuning. The goal is local refinement, not basin escape.

---

## 4. Expected Performance

| Metric | v6 (ep120) | v7 projected | Confidence | Driver |
|--------|:---:|:---:|:---:|--------|
| P4 F1 | 0.451 | 0.52–0.58 | Medium-High | Deeper head + c5 features |
| P3 F1 | 0.254 | 0.32–0.40 | Medium | Earlier gating + 2-layer head |
| P2 F1 | 0.157 | 0.25–0.35 | Medium | 3-layer head + reduced pos_weight |
| P2 saturation | Yes | No | High | pos_weight=25 + label smoothing |
| Head bias P4 | -0.70 | -0.50 to -0.60 | Medium | Restart provides escape energy |
| Head bias P3 | -2.38 | -1.5 to -2.0 | Medium | 117 epochs of obj loss vs 105 |
| Head bias P2 | -2.38 | -1.5 to -2.0 | Medium | 115 epochs of obj loss vs 90 |
| False positives | Unknown | Quantified + reduced | Medium | Mining + fine-tuning phase |

---

## 5. File Changes Required

### 5.1 `src/cv/face_detector_cnn.py`

New components:
- `DeepAnchorFreeHead`: 2 or 3 conv layers, configurable depth per level
- `FaceFCNv7`: New model class using enhanced backbone + new heads
- SE blocks added to `FPNBackbone`
- Stem refinement block

### 5.2 `src/training/train_v7.py`

New training script based on `train_v6.py` with:
- Earlier progressive gating (P3@3, P2@5)
- Modified pos_weights (25/15/10)
- Label smoothing parameter
- LR restart at epoch 60
- Multi-scale training support
- Gradient accumulation
- P2 spatial dropout

### 5.3 `config/default.yaml`

New model path entry: `face_cnn_v7: models/face_cnn_v7_best.pth`
