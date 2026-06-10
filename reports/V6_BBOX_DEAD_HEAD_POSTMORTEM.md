# FaceCNN v6.0 — Post-Mortem: The Dead Bbox Head Failure

**Date:** May 27, 2026
**Status:** Documenting the root cause of v6's 0.44% detection recall

---

## 1. The False Metric: Why Heatmap F1=0.319 Lied

Throughout v6 training, the validation F1 rose steadily to 0.319 at epoch 56.
This was treated as a success — the model was "learning." In reality, the
heatmap F1 metric was measuring something unrelated to actual detection quality.

### 1.1 What Heatmap F1 Actually Measures

The validation function in `train_v6.py:298-342` computes F1 by comparing
every cell in the objectness heatmap against the ground-truth heatmap at
threshold 0.5:

```python
pred_probs = torch.sigmoid(pred_obj)       # [B, 1, H, W]
gt_bin = (t_hm > 0.5).float()              # ground truth heatmap
pred_bin = (pred_probs > 0.5).float()      # prediction at 0.5
tp = (pred_bin * gt_bin).sum()             # cell matches
fp = (pred_bin * (1 - gt_bin)).sum()
fn = ((1 - pred_bin) * gt_bin).sum()
```

This is a **per-cell binary classification** metric. It asks: "Is each
individual cell predicting face or background?" At epoch 56, 31.9% of cells
were correct.

### 1.2 What It Does NOT Measure

The heatmap F1 does NOT measure:
- Whether the peak-finding (dilation + local maxima) finds anything
- Whether the bbox decode produces boxes in the right location
- Whether the bbox decode produces boxes at the right size
- Whether any detection matches a real face at IoU ≥ 0.5

These are entirely separate from per-cell classification. The model could
predict every cell correctly (heatmap F1=1.0) and still produce zero usable
detections if the bbox regression heads were dead.

### 1.3 The Discrepancy

| Metric | Epoch 56 Value | What It Actually Measures |
|--------|:---:|------|
| Heatmap P4 F1 | 0.468 | P4 cells predict face vs background correctly 46.8% of the time |
| Heatmap P3 F1 | 0.285 | P3 cells predict face vs background correctly 28.5% of the time |
| **Detection recall** | **0.0044** | **0.44% of GT faces are detected by the full pipeline** |

The 106× gap between heatmap F1 (0.319) and detection recall (0.0044) is the
cost of the broken bbox pipeline.

---

## 2. The Diagnosis: Dead P3/P4 Bbox Heads

### 2.1 Head Weight Collapse

On the epoch 56 checkpoint, the bbox heads had the following weight statistics:

| Head | Parameter Count | Weight L2 Norm | Std | Expected L2 (kaiming init) | Status |
|------|:---:|:---:|:---:|:---:|:---:|
| head_p2.bbox_pred | 260 | **2.2766** | 0.1414 | ~2.0 | ✓ **Alive** |
| head_p2.obj_pred | 65 | 8.8878 | — | — | ✓ **Alive** |
| head_p3.bbox_pred | 260 | **0.0506** | 0.0032 | ~2.0 | ✗ **Dead (45× below init)** |
| head_p3.obj_pred | 65 | 10.0101 | — | — | ✓ **Alive** |
| head_p4.bbox_pred | 260 | **0.0355** | 0.0022 | ~2.0 | ✗ **Dead (56× below init)** |
| head_p4.obj_pred | 65 | 5.8866 | — | — | ✓ **Alive** |

Key observation: ALL obj_pred heads are healthy (5.9-10.0 L2). The objectness
classification worked. Only the bbox_pred heads at P3 and P4 collapsed.

### 2.2 What Dead Means in Practice

With bbox weights at L2=0.05 (near zero), the bbox output is essentially the
bias plus random noise from near-zero weights. The bias init was 0.0 for all
outputs:

```
head_p3.bbox_pred.bias = [0.00015, 3.5e-5, -0.00041, -0.00051]
```

The decoded bbox formula:
```python
dx = output[0]           # ~0.00015 → center offset ~0 pixels
dy = output[1]           # ~3.5e-5  → center offset ~0 pixels
dw = output[2]           # ~-0.0004 → exp(-0.0004) ≈ 0.9996
dh = output[3]           # ~-0.0005 → exp(-0.0005) ≈ 0.9995
box_w = exp(clip(dw)) * stride  # ≈ 1.0 * 4 = 4 pixels at P3
```

At stride 4 (P3), the default box is ~4 pixels. At stride 8 (P4), it's ~8
pixels. These boxes are filtered by the minimum size check (5px), so almost
all P3/P4 detections are excluded by the size filter.

This is why detection recall is 0.44% — the model never produces boxes that
survive the inference pipeline.

### 2.3 The Gradient Starvation Mechanism

The bbox heads collapsed because they received insufficient gradient signal
during training. The loss computation in `train_v6.py`:

```python
for level in ['p2', 'p3', 'p4']:
    obj_l = criterion_cls[level](pred_obj, t_hm)           # BCE loss
    if pos_mask.sum() > 0:
        bbox_l = criterion_bbox(pred_bbox, t_bb, pos_mask)  # GIoU loss
    loss = loss + obj_l + bbox_l
```

At epoch 56, the loss components were:

| Level | Obj Loss | Bbox Loss | Obj/Bbox Ratio |
|-------|:---:|:---:|:---:|
| P2 | 0.0620 | 0.0473 | **1.3×** |
| P3 | 0.1026 | 0.0472 | **2.2×** |
| P4 | 0.8092 | 0.0124 | **65.3×** |
| **Total** | **0.9738** | **0.1069** | **9.1×** |

The obj loss dominates the total loss by 9.1×. The bbox loss receives ~10%
of the total gradient signal. For P4 specifically, obj loss is 65× larger
than bbox loss.

When AdamW applies weight updates, the bbox parameters receive only ~2-10%
of the gradient that obj parameters receive. Over 120 epochs, the bbox weights
drift toward zero while obj weights grow to 6-10 L2.

### 2.4 Why P2 Escaped While P3/P4 Collapsed

P2's bbox head (L2=2.28) survived because P2 had more bbox training signal:

| Property | P2 | P3 | P4 |
|----------|:---:|:---:|:---:|
| Stride | 2 | 4 | 8 |
| Grid size | 320×240 | 160×120 | 80×60 |
| GT face area per cell | 2×2 px | 4×4 px | 8×8 px |
| Avg positive cells per image | ~40 | ~10 | ~3 |
| Bbox loss per epoch | 0.047 | 0.047 | 0.012 |

P2 has larger feature maps (320×240 = 76,800 cells) where faces occupy more
cells on average (~40 vs ~3-10). Each bbox update at P2 sees more training
examples, making the gradient more robust.

P4 has the smallest grid (80×60 = 4,800 cells) with only ~3 positive cells
per image. The positive-to-negative ratio is 3:4,797 ≈ 1:1,600. At this
ratio, the GIoU loss is applied so rarely that the bbox weights never
accumulate meaningful gradient.

### 2.5 The Degenerate FP Behavior

With near-zero output, the bbox "giant box" behavior emerged:

```
Source: The 30,104 FPs are ALL from P3.
Source: Every FP box is ~593 pixels wide.
```

A dead P3 bbox head outputs dw≈5 for every peak. Why?

The answer: when weights are near-zero, the AdamW optimizer state is also
near-zero. If a small gradient pushes the output slightly positive,
AdamW's momentum and adaptive LR can accelerate it to +5 in a few updates.
The clip at +5 then holds it there permanently — AdamW has no mechanism
to pull it back down from the clipping boundary.

This creates a self-reinforcing cycle:
1. Bbox output drifts positive
2. Clip(·, -2, +5) caps it at +5
3. Gradient at +5 is zero (clipping kills gradient)
4. AdamW's state decays toward zero
5. Bbox output stays at +5 forever

Result: every P3 peak produces a box of exp(5)×4 = 593 pixels — the
maximum possible, covering the entire image.

---

## 3. System-Wide Impact

### 3.1 False Positive Cascade

The 30,104 FPs all come from P3's giant-box failure. P2 produces no FPs
because its bbox head works (reasonable boxes) but its obj head is
saturated (100% cells fire). P4 produces no FPs because its near-zero
bbox output produces tiny boxes that are filtered by the size check.

The effective detection count at the pipeline output: 16 TPs out of 3,613
ground truth faces across 500 images.

### 3.2 Comparison to v5 Dead Head

v5 had the opposite problem — dead OBJ head (bias -4.6, sigmoid 0.01).
v6 fixed the obj head (bias -0.80, sigmoid 0.31) but introduced dead
BBOX heads.

| Iteration | OBJ | BBOX | Result |
|-----------|:---:|:----:|--------|
| v5 | ✗ Dead (bias -4.6) | Unknown | 0 detections |
| v6 | ✓ Alive (bias -0.80) | ✗ **Dead (L2=0.04)** | 16 TPs / 30K FPs |
| **v7** | ✓ **Multi-layer** | ✓ **4-layer + bbox weight** | Target: ~50% recall |

### 3.3 Why Heatmap F1 Was Not the Right Metric

The heatmap F1 tracked in the training CSV uses threshold 0.5 on sigmoid
output. At this threshold, a cell either "fires" or doesn't. The obj heads
were well-calibrated, so the cell-level binary accuracy was meaningful for
objectness. But objectness alone doesn't detect faces — boxes must be
correctly located and sized.

The correlation between heatmap F1 and detection quality is ZERO when the
bbox heads are dead. The training metric should track bbox head health,
not just obj F1.

---

## 4. Lessons Learned for v7

| Lesson | Root Cause | v7 Fix |
|--------|-----------|--------|
| Single 1×1 conv head cannot learn both obj and bbox | 65 params total — no non-linear capacity | **4-layer head** (116K params/level) |
| Bbox loss gets 10% of gradient — starves | Obj:bbox ratio ~9:1 | **Bbox weight multiplier (5×)** in training loss |
| Bbox bias at 0.0 means default box is 0 pixels | No useful starting point | **Bbox bias = -2.0** → default box = exp(-2)×stride |
| No bbox health metric in training CSV | Only F1 tracked | **Track bbox weight L2 norm** per epoch |
| P4 has 3:4,797 positive ratio | Almost no gradient visits P4 | **Drop P2, keep P3/P4** (better positive ratios) |
| No structural prevention of weight collapse | AdamW can accelerate any output | **Multi-layer heads** with BN → gradient flows through all layers |
| Clip(dw, -2, +5) creates dead zone at +5 | No gradient signal at boundary | **Wider clip range** (-4, +8) combined with proper init |

---

*Document Version: 1.0 — May 27, 2026*
