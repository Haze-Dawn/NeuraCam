# FaceCNN v8.0 — Architecture, Training & Full Optimization Pipeline

**Date:** June 1, 2026
**Status:** Design phase — not yet implemented
**Target:** 0.72-0.78 mAP on WIDER Face val, <35ms CPU inference per frame

---

## Table of Contents

1. [Architecture Design](#1-architecture-design)
   - 1.1 [Lessons from V7](#11-lessons-from-v7)
   - 1.2 [Backbone](#12-backbone)
   - 1.3 [BiFPN Neck](#13-bifpn-neck)
   - 1.4 [Independent Detection Heads](#14-independent-detection-heads)
   - 1.5 [Total Parameter Budget](#15-total-parameter-budget)
   - 1.6 [Why Independent Heads Fix P3](#16-why-independent-heads-fix-p3)
2. [Training Pipeline](#2-training-pipeline)
   - 2.1 [Loss Functions](#21-loss-functions)
   - 2.2 [Label Assignment Strategy](#22-label-assignment-strategy)
   - 2.3 [Optimizer and Schedule](#23-optimizer-and-schedule)
   - 2.4 [Data Augmentation](#24-data-augmentation)
   - 2.5 [Hard-Aware Sample Redistribution](#25-hard-aware-sample-redistribution)
   - 2.6 [Training Configuration](#26-training-configuration)
   - 2.7 [6GB VRAM Management](#27-6gb-vram-management)
   - 2.8 [Monitoring and Diagnostics](#28-monitoring-and-diagnostics)
3. [Data Expansion and Distillation](#3-data-expansion-and-distillation)
   - 3.1 [Baseline Pseudo-Labeling (3-Cycle)](#31-baseline-pseudo-labeling-3-cycle)
   - 3.2 [WIDER Training Set Re-Labeling](#32-wider-training-set-re-labeling)
   - 3.3 [Cross-Dataset Pseudo-Labeling](#33-cross-dataset-pseudo-labeling)
   - 3.4 [Noisy Student Training](#34-noisy-student-training)
   - 3.5 [SCRFD Teacher Distillation](#35-scrfd-teacher-distillation)
   - 3.6 [Curriculum Threshold Scheduling](#36-curriculum-threshold-scheduling)
4. [Post-Training Inference Optimizations](#4-post-training-inference-optimizations)
   - 4.1 [Per-Level Threshold Calibration](#41-per-level-threshold-calibration)
   - 4.2 [Test-Time Flip Averaging](#42-test-time-flip-averaging)
   - 4.3 [Weighted Box Fusion (WBF)](#43-weighted-box-fusion-wbf)
   - 4.4 [DIoU-NMS / Adaptive NMS](#44-dious-nms--adaptive-nms)
   - 4.5 [Multi-Scale Testing](#45-multi-scale-testing)
   - 4.6 [Quality Score α/β Calibration](#46-quality-score-αβ-calibration)
   - 4.7 [Temperature Scaling](#47-temperature-scaling)
   - 4.8 [FPN Cross-Level Box Voting](#48-fpn-cross-level-box-voting)
   - 4.9 [Tiling for High-Resolution Inputs](#49-tiling-for-high-resolution-inputs)
   - 4.10 [Model Soup (Weight Averaging)](#410-model-soup-weight-averaging)
   - 4.11 [Detection Ensemble (Multi-Checkpoint)](#411-detection-ensemble-multi-checkpoint)
5. [CPU Deployment](#5-cpu-deployment)
   - 5.1 [ONNX Export and INT8 Quantization](#51-onnx-export-and-int8-quantization)
   - 5.2 [OpenVINO Optimization](#52-openvino-optimization)
   - 5.3 [NCNN Deployment (Optional)](#53-ncnn-deployment-optional)
   - 5.4 [GPU Inference via TensorRT](#54-gpu-inference-via-tensorrt)
   - 5.5 [Target Benchmarks](#55-target-benchmarks)
6. [Projected Metrics and Roadmap](#6-projected-metrics-and-roadmap)
   - 6.1 [Phase Plan](#61-phase-plan)
   - 6.2 [Cumulative mAP Projection](#62-cumulative-map-projection)
   - 6.3 [Success Criteria](#63-success-criteria)
   - 6.4 [Risks and Mitigations](#64-risks-and-mitigations)

---

## 1. Architecture Design

### 1.1 Lessons from V7

| Problem | V7 Behavior | V8 Fix |
|---------|-------------|--------|
| P3 gradient cancellation | Shared conv1/conv2 across levels → face cells cancel against background | **Independent per-level heads** — each head trains only on its own stride's positives |
| Precision bottleneck (0.45) | Fixed Gaussian assignment, BCE loss | **Varifocal + EIoU + sample-quality consistent soft labels** |
| Single detection level | P4-only (stride 8) misses sub-35px faces | **3 FPN levels** (stride 4/P3, stride 8/P4, stride 16/P5) |
| Manual backbone suboptimal | 16 DSConv blocks, max 128ch, manual depth | **Empirically optimized depth allocation** — deeper stages 2-3, max 192ch |
| Simple FPN addition | Element-wise add of lateral + top-down | **BiFPN** with learned per-channel weighted fusion |
| 217ms CPU inference | 494K params, vanilla FPN | **ONNX INT8 + OpenVINO → ~30ms** |
| Insufficient training data | 12,880 WIDER training images | **5.7× data expansion** via pseudo-labeling + cross-dataset |

### 1.2 Backbone

Depthwise separable conv blocks throughout, using an empirically tuned depth and width distribution optimized for face detection at multiple scales:

```
Input: 3×480×640

Stem: Conv3×3 s=2  (3→32) → BN → HardSwish
  Output: 32×240×320 | RF: 3×3 | Params: 896

Stage 1: 3× DSConv (32→64→64)
  Block 1: stride=2 (downsample: 32→64)
  Blocks 2-3: stride=1 (refine: 64→64)
  Output: 64×120×160 | RF: 10×10 | Params: 6,528
  → C2 at stride 2 (used by FPN for context only)

Stage 2: 6× DSConv (64→128→128→128→128→128→128)
  Block 1: stride=2 (downsample: 64→128)
  Blocks 2-6: stride=1 (refine: 128→128)
  Output: 128×60×80 | RF: 30×30 | Params: 49,152
  ← C3 at stride 4 (FPN level P3 input)

Stage 3: 6× DSConv (128→160→160→160→160→160→160)
  Block 1: stride=1 (keep stride 4, expand: 128→160)
  Blocks 2-6: stride=1 (refine: 160→160)
  Output: 160×60×80 | RF: 42×42 | Params: 61,440
  ← C4 at stride 4 (FPN level P4 input — upsampled)

Stage 4: 6× DSConv (160→192→192→192×4)
  Block 1: stride=2 → dilation=1 (downsample: 160→192, 30×40)
  Blocks 2-6: stride=1, dilation=2 (refine: 192×5, RF grows via dilation)
  Output: 192×30×40 | RF: 78×78 | Params: 73,728
  ← C5 at stride 8 (FPN top-down source + P5 input)

Total backbone: ~191K params
```

**Key decisions:**
- **Max 192ch** — 33% wider than V7's 128ch. Cost: each DSConv block jumps from 18K to 25K params. But with **fewer blocks in stage 1 and more in stages 2-3** (where face features matter), the total stays manageable.
- **HardSwish activation** — `x * sigmoid(x)` gives 0.1-0.3% better accuracy than ReLU at zero extra cost (fused in ONNX/OpenVINO as single op).
- **No pooling layers** — all downsampling via strided convs (learned, not fixed).
- **Stage 3 kept at stride 4** (not stride 8) — unlike typical ImageNet backbones that downsample to stride 8 at this stage. Face detection needs high-resolution features for small faces; keeping stride 4 longer preserves spatial detail for P3.
- **Stage 4 dilation=2** — expands RF from 42×42 to 78×78 without losing resolution. Critical for large-face context at P5.

### 1.3 BiFPN Neck

Replaces V7's simple top-down addition with **weighted feature fusion** across P3, P4, P5, with extra P2 (context) and P6 (global) levels:

```
C2 (stride 2,  64ch,  240×320)  ──lat2──→ P2_OUT  (context only)
C3 (stride 4, 128ch,  120×160)  ──lat3──→ P3_IN             
C4 (stride 4, 160ch,   60×80)   ──lat4──→ P4_IN             
C5 (stride 8, 192ch,   30×40)   ──lat5──→ P5_IN             
```

**Top-down pass** (semantic context flows down):
```
P5 = lat5(C5)                                                    → 96×30×40
P4 = fuse(lat4(C4), upsample(P5))                                 → 96×60×80
P3 = fuse(lat3(C3), upsample(P4))                                 → 96×120×160
P2 = fuse(lat2(C2), upsample(P3))                                 → 96×240×320
```

**Bottom-up pass** (spatial detail flows up — the PANet addition):
```
P2_out = P2                                                       → 96×240×320
P3_out = fuse(P3, downsample(P2_out))                              → 96×120×160
P4_out = fuse(P4, downsample(P3_out))                              → 96×60×80
P5_out = fuse(P5, downsample(P4_out))                              → 96×30×40
P6_out = P5_out → Conv3×3 s=2 stride-2 → (global context)         → 96×15×20
```

**Weighted fusion** replaces element-wise add with learned per-channel weights:

```python
def fuse(feats):
    # feats = [feat_a, feat_b]  # lateral + upsampled
    w = softmax(learned_weights)  # 2 per-channel weights, per-level
    return w[0] * feats[0] + w[1] * feats[1]
```

Each fusion node learns 2×96 = 192 weights per level. Inserts also a 3×3 depthwise-separable refine conv (same as V7's refine3/refine4) after each fusion to remove aliasing.

**Detection levels (what heads see):**
- **P3** (stride 4,  120×160,  96ch) — small faces, 16-64px
- **P4** (stride 8,   60×80,   96ch) — medium faces, 32-128px
- **P5** (stride 16,  30×40,   96ch) — large faces, 64-256px

**Total FPN params: ~118K**

### 1.4 Independent Detection Heads

**NO shared weights between levels.** Each head trains exclusively on its own stride's feature map and its own positive cells. This is the architectural fix for the gradient cancellation problem that killed P3 in V7.

Each head is a 3-layer MLP with 1×1 convs (no 3×3 spatial mixing — the BiFPN already provides spatial context):

```
Head architecture (all levels share this design, but separate weight tensors):

P3_feat (96ch, 120×160) 
  → conv1: Conv2d(96→96, 1) → BN → HardSwish
  → conv2: Conv2d(96→96, 1) → BN → HardSwish
  → obj:   Conv2d(96→1,  1)  → logit per cell
  → iou:   Conv2d(96→1,  1)  → IoU prediction per cell
  → bbox:  Conv2d(96→4,  1)  → (dx, dy, dw, dh) per cell

P4 head: same design, P4_feat (96ch, 60×80) as input
  → 3× same conv structure, separate weights

P5 head: same design, P5_feat (96ch, 30×40) as input
  → 3× same conv structure, separate weights
```

**Params per level:**
| Layer | Input→Output | Weight Params | BN Params | Total |
|-------|:-----------:|:-------------:|:---------:|:-----:|
| conv1 | 96→96 | 9,216 | 192 | 9,408 |
| conv2 | 96→96 | 9,216 | 192 | 9,408 |
| obj | 96→1 | 96 | — | 96 |
| iou | 96→1 | 96 | — | 96 |
| bbox | 96→4 | 384 | — | 384 |
| **Per level** | | | | **19,392** |
| **3 levels total** | | | | **58,176** |

**Bias initialization (per level):**
```python
nn.init.constant_(head.obj.bias, -3.0)    # sigmoid(-3.0) = 0.047
nn.init.constant_(head.iou.bias, 0.0)      # sigmoid(0.0) = 0.5 (uncertain)
nn.init.zeros_(head.bbox.bias)
head.bbox.bias.data[2:] = -2.0             # dw, dh = -2 (start small)
```

P3 bias=-3.0 (vs V7's -2.5) because stride 4 has 4× more cells than stride 8 — lower prior needed to compensate for denser grid. P4/P5 use -2.5.

### 1.5 Total Parameter Budget

| Component | Params | % of Total | Notes |
|-----------|:------:|:----------:|-------|
| Backbone | 191,264 | 25% | DSConv, max 192ch, 21 blocks |
| FPN (lat + BiFPN + refine + SE) | 117,648 | 15% | 6 levels (P2-P6), weighted fusion |
| **Detection heads** (×3, independent) | **58,176** | **8%** | P3/P4/P5, 3-layer MLP each |
| Projection layers (lat convs) | 48,128 | 6% | 5 lateral 1×1 convs (C→96ch) |
| **Total** | **760,512** | | |
| *SCRFD-0.5GF for comparison* | *570K* | | *NAS-optimized* |

We're ~33% larger than SCRFD-0.5GF, but all manual design. The extra ~190K params are in the backbone (wider stages) and BiFPN (bottom-up path + weighted fusion), which are both well-motivated design choices independent of SCRFD.

### 1.6 Why Independent Heads Fix P3

V7's shared conv1/conv2 (166K params) processed both P3 and P4 features through the SAME weights. The gradient was:

```
dL/dw = Σ_{P3 cells} dL/d(cell) + Σ_{P4 cells} dL/d(cell)
```

With 19,200 P3 cells (600 face, 18,600 bg) + 4,800 P4 cells (40 face, 4,760 bg) = 24,000 total cells feeding the same weights, the face-cell gradient was drowned by background. Worse, the 3×3 conv summed across these 24,000 positions — face cells wanting sigmoid↑ against background wanting sigmoid↓ produced near-zero net gradient.

**With independent heads:**
- P3 head trains exclusively on its 19,200 cells, 600 of which are ATSS-positive
- P4 head trains exclusively on its 4,800 cells, 40 of which are ATSS-positive
- Each head's weights only see gradient from its own stride
- No cancellation — P3 face gradient (600 × 15 pos_weight) dominates P3's 18,600 bg gradient when combined with OHEM

Each level still has the class imbalance problem, but OHEM (keep only top-5× hardest negatives per positive) solves it at the head level rather than at the shared-weight level. The key difference: **each head can be independently tuned** — P3 can have pos_weight=15, neg_ratio=5, while P4 uses pos_weight=10, neg_ratio=3.

---

## 2. Training Pipeline

### 2.1 Loss Functions

#### 2.1.1 Varifocal Loss (Objectness + IoU Quality)

Replaces BCE + IoU branch with a unified loss. VarifocalLoss treats the obj prediction as a **continuous IoU quality score** rather than a binary face/background classifier:

```
VFL(p, t) = 
  -t × (1-p)^γ × log(p)           if target t > 0 (foreground)
  -(1-t) × p^γ × log(1-p)         if target t = 0 (background)
```

Where:
- `p` = predicted quality score (sigmoid of obj logit)
- `t` = target quality score (IoU of predicted bbox with GT, clamped to [0,1])
- `γ` = 2.0 (focusing parameter)

**Why VarifocalLoss over BCE:**
- **Foreground**: gradient is `(1-p)^γ × (p - t)` — proportional to (1-p)^γ, highest when prediction is wrong. Unlike BCE where gradient is (p - t) regardless of confidence.
- **Background**: gradient is `p^γ × p` — only present when the model incorrectly predicts a high quality score. Correct background predictions (p≈0) contribute near-zero gradient, just like OHEM.
- **Continuous targets**: each positive cell has a unique target = its actual IoU with GT. No binary 0/1 — no degenerate fixed point.
- **Self-calibrating**: the model learns to predict "how well does this cell localize" rather than "is this a face." The final quality score is the predicted IoU, which directly feeds into NMS ranking.

#### 2.1.2 EIoU Loss (Bounding Box Regression)

Replaces V7's CIoU with EIoU. CIoU's aspect ratio term is redundant with center distance. EIoU simplifies and improves:

```
EIoU = IoU - ρ²(c, c_gt)/c² - ρ²(w, w_gt)/C_w² - ρ²(h, h_gt)/C_h²

Loss = 1 - EIoU
```

Where:
- `ρ²(c, c_gt)` = squared center distance
- `ρ²(w, w_gt)` = squared width difference
- `ρ²(h, h_gt)` = squared height difference
- `C_w, C_h` = width/height of the smallest enclosing box

**Why EIoU over CIoU:**
- CIoU's aspect ratio term `αv` doesn't reduce width/height difference directly — it only reduces the ratio mismatch. EIoU directly penalizes w/h errors.
- Faster convergence for bbox regression (validated in SCRFD and YOLOX).
- All three terms have non-zero gradient for ALL overlaps — no dead zone.

#### 2.1.3 Distribution Focal Loss (DFL)

Predict bbox offsets as a distribution over discrete values rather than a single scalar:

```python
# Instead of predicting dx, dy, dw, dh directly:
# Predict 16-bin distribution for each of 4 bbox parameters
dfl_pred = Conv2d(96, 4 × 16, 1)  # 4 offsets × 16 bins each
dx = sum(softmax(dfl_pred[:, 0:16]) × [0, 1, ..., 15] / 7.5 - 1)
```

**Why DFL:** Gives 4× more capacity to bbox regression at the cost of 4× the output channels (384 vs 96 for the bbox head). The distribution captures uncertainty — a face with ambiguous edge creates a wide distribution, while a clearly bounded face creates a sharp peak. DFL alone adds ~1-2 mAP on COCO (proven in VarifocalNet, GFL).

**Total head output:** obj(1ch) + iou(1ch) + bbox_dfl(64ch = 4×16) = 66 channels per level (vs 6 in V7). The extra 60 channels add 60 × 96 = 5,760 params per head, or ~17K total across 3 levels — negligible.

#### 2.1.4 Loss Weighting

```
Total Loss = 1.0 × VarifocalLoss(obj) + 1.0 × EIoU_Loss(bbox) + 0.5 × MSE(iou_pred, actual_iou)
```

The iou MSE branch has 0.5 loss weight because it trains on a simpler task (regress a single scalar) and converges faster. All three losses are trained simultaneously from epoch 1 (no delayed head activation — each head starts training immediately since heads are small and converge quickly).

### 2.2 Label Assignment Strategy

#### 2.2.1 Sample-Quality Consistent Assignment

Each positive cell is assigned a **soft quality score** rather than a hard binary label:

```python
for each GT face:
    # 1. Compute IoU of each cell's decoded bbox with this GT
    # 2. Take top-k (k=9) IoU values
    # 3. Compute threshold = mean(topk) + std(topk)
    # 4. Cells with IoU > threshold are positive
    # 5. POSITIVE TARGET = IoU_value (not 1.0) → quality-neutral label
    # 6. NEGATIVE TARGET = 0.0 → still binary, but positives are fractional
```

This differs from ATSS in step 5: ATSS uses 1.0 for all positives. Sample-quality consistent assignment uses the **actual IoU** as the target. This means:
- A cell at the face center (IoU=0.85) targets 0.85
- A cell at the face edge (IoU=0.35) targets 0.35
- A background cell targets 0.0

**Why this helps:** The model learns to predict the IoU quality score directly, not "is this a face." This makes the quality score at inference well-calibrated by design — it IS the predicted localization quality. No calibration needed.

#### 2.2.2 Per-Level Positive Count Targets

| Level | Stride | Grid Cells | Avg Positives/Image | Pos:Neg (raw) | Pos:Neg (after OHEM) |
|-------|:------:|:----------:|:-------------------:|:-------------:|:--------------------:|
| P3 | 4 | 19,200 | ~80 | 1:240 | **1:5** |
| P4 | 8 | 4,800 | ~40 | 1:120 | **1:5** |
| P5 | 16 | 1,200 | ~20 | 1:60 | **1:5** |

The OHEM ratio of 5:1 negatives to positives is maintained per-level, per-image. This ensures each level receives balanced gradient regardless of the raw class imbalance.

### 2.3 Optimizer and Schedule

#### 2.3.1 Optimizer

**AdamW** with decoupled weight decay:
- `lr = 5e-3` (head layers), `lr = 1e-3` (backbone — already well-initialized)
- `weight_decay = 0.05` (per SCRFD — stronger than typical 1e-4)
- `betas = (0.9, 0.999)`
- `eps = 1e-8`
- Parameter group separation:
  - Backbone: lr 1e-3, wd 0.05
  - FPN + BiFPN: lr 2e-3, wd 0.05
  - Detection heads: lr 5e-3, wd 0.0 (heads need no regularization)
  - Bias terms: lr 5e-3, wd 0.0 (no weight decay on biases)

#### 2.3.2 Schedule

```
Epochs 1-5:   Warmup (linear 1e-6 → full LR per group)
Epochs 5-70:  Cycle 1 — CosineAnnealing (full LR → 1e-5)
Epochs 71-135: Cycle 2 — SGDR restart at full LR
Epochs 136-200: Cycle 3 — SGDR restart at full LR
Epochs 201-300: Extended fine-tuning with LR 1e-5 → 1e-6 (cosine)
```

**Why 300 epochs (vs V7's 150):** 
- 3 independent heads need time to converge (each head starts from scratch)
- Sample redistribution means harder images are over-represented — convergence is slower
- SGDR restarts at epochs 70 and 135 escape local minima in head training
- Extended fine-tuning (epochs 201-300) at low LR is optional — adds ~0.01-0.02 mAP by slowly refining features on the hardest images

#### 2.3.3 Batch Size and Gradient Accumulation

- **Batch size = 8** (stable on RTX 2060 6GB, same as V7's proven config)
- **Gradient accumulation = 4** (effective batch = 32)
- `torch.cuda.empty_cache()` at epoch boundaries
- BN running stats updated after each epoch's accumulation step

### 2.4 Data Augmentation

#### Already in V7 (carried forward):

| Augmentation | Probability | Config | Purpose |
|-------------|:-----------:|--------|---------|
| Multi-scale resize | 100% | Random 384-800px short side, then crop to 480×640 | Scale invariance |
| Copy-paste | 50% per batch | Paste 5 random face crops from batchmates | Improves pos:neg ratio by 3× |
| Horizontal flip | 50% per image | Mirror image + flip bbox coords | Standard |

#### New in V8:

| Augmentation | Probability | Config | Purpose |
|-------------|:-----------:|--------|---------|
| **RandAugment** | 50% per batch | N=2, M=9 (strong) | Color + contrast + translation invariance |
| **GridMask** | 25% per image | Ratio=0.4, rotate=1 | Occlusion robustness — forces model to detect faces through partial obstruction |
| **MixUp** | 30% per batch | α=0.2, β=0.2 | Blends images + targets → softer labels, better calibration |
| **CutMix** | 20% per batch | Replace random square region | Localized MixUp — keeps most of image intact but forces attention |
| **HSV jitter** | 80% per image | H:±30°, S:±30%, V:±30% | Lighting variation (makes backbone invariant to color casts) |
| **Blur** | 10% per image | Gaussian blur σ=0.5-1.5 | Motion blur robustness |

**Important:** Augmentation strength on pseudo-labeled images is 1.5× higher than on labeled images (Noisy Student principle — student must work harder on the same targets).

### 2.5 Hard-Aware Sample Redistribution

After epoch 50, compute per-image difficulty score = 1 - max(quality score for any correct detection on that image). Images where the model struggles (no correct detections, or very low quality) are **resampled at 2× frequency**.

```python
# After epoch 50:
for epoch in range(51, 301):
    difficulty = compute_per_image_difficulty(model, val_loader)
    sample_weight = 1.0 + difficulty  # range: 1.0 (easy) to 2.0 (hard)
    sampler = WeightedRandomSampler(sample_weight, replacement=True)
    train_loader = DataLoader(..., sampler=sampler)
```

**Why this helps:** The hardest images in WIDER Face (crowded scenes, severe occlusion, tiny faces) are underrepresented in uniform sampling. Each epoch shows ~3,500 unique images. Redistribution ensures hard images appear 2× more often, giving the model more gradient signal on its failure modes.

**Cost:** Adds ~10% per-epoch time (recomputation of sampler). Negligible.

### 2.6 Training Configuration

```bash
python3 src/training/train_v8.py \
  --data data/face/widerface \
  --output models/face_cnn_v8.pth \
  --epochs 300 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr-backbone 1e-3 \
  --lr-head 5e-3 \
  --weight-decay 0.05 \
  --sgdr-t0 65 \
  --smoothing 0.1 \
  --varifocal-gamma 2.0 \
  --eiou-loss \
  --dfl-bins 16 \
  --atss \
  --multi-scale 384 800 \
  --copy-paste 5 \
  --randaugment 2 9 \
  --gridmask 0.4 \
  --mixup 0.2 \
  --hsv-jitter 0.3 \
  --blur 0.1 \
  --hard-redistribute 50 \
  --ckpt-interval 5 \
  --diag-interval 1 \
  --validate-interval 5
```

### 2.7 6GB VRAM Management

From V7 experience, batch-8 with gradient accumulation 4 is stable. Additional measures:

| Measure | Why |
|---------|-----|
| `torch.cuda.empty_cache()` at start of each epoch | Prevents fragmentation from building up |
| BN running_var clamp (min=1e-4) | Prevents NaN at batch-8 (proven fix from V7 §5.3) |
| Gradient checkpointing in backbone (stage 3-4 only) | Saves ~400MB VRAM at cost of ~15% slower backward pass |
| FP32 training (no AMP) | AMP's gradient scaling interacts poorly with VarifocalLoss's asymmetric terms. ~200MB extra cost accepted. |
| Detach pseudo-label targets before passing to loss | Prevents gradient flow through the teacher model |

**Peak VRAM: ~5.2 GiB** (within the 6GB budget, leaving 800MB headroom for augmentation intermediates).

### 2.8 Monitoring and Diagnostics

Per-epoch diagnostics (same pattern as V7, saved to `v8_diagnostics/`):

| Metric | What It Catches |
|--------|-----------------|
| Per-level obj bias (sigmoid) | Is a head saturating? |
| Per-level bbox L2 norm | Is a bbox head dying? |
| Per-level F1 at threshold 0.5 | Is the head producing useful detections? |
| Per-level recall at threshold 0.05 | Is the head firing at all? |
| Gradient norm (total + per-group) | Is training progressing or stuck? |
| BN running_mean/var stats | Are stats evolving normally? |
| Per-image difficulty distribution | Is redistribution working? |
| Number of ATSS positives per level per image | Is assignment producing enough positives? |

---

## 3. Data Expansion and Distillation

### 3.1 Baseline Pseudo-Labeling (3-Cycle)

After initial 300-epoch training, run 3 cycles of pseudo-labeling on the 16,103 unlabeled WIDER test images:

| Cycle | Pseudo-Label Source | Total Training Data | Epochs | Expected mAP Gain |
|:-----:|:-------------------:|:------------------:|:------:|:-----------------:|
| 1 | V8 base → 5K highest confidence detection | 12,880 + 5K = 17,880 | 25 | +0.03 |
| 2 | Cycle 1 model → 10K detections | 12,880 + 10K = 22,880 | 25 | +0.02 |
| 3 | Cycle 2 model → 16K detections | 12,880 + 16K = 28,880 | 25 | +0.01 |

**Total data: 2.3× base. Total time: ~5h.**

**Key parameters:**
- Quality threshold: 0.4 (stricter than V7 plan's 0.3 — better model = higher bar)
- Label smoothing: s=0.2 (accounts for pseudo-label noise)
- Each cycle is a fine-tune of the previous cycle's best model at LR 1e-4 → 1e-6 (cosine)
- Mining active throughout — any false positives in pseudo-labels are caught and corrected

### 3.2 WIDER Training Set Re-Labeling

WIDER Face annotations are incomplete — 15-20% of faces (especially small/occluded/boundary faces) are unlabeled. After V8 base training:

1. Run inference on all 12,880 training images
2. Collect detections with quality > 0.4 that have IoU < 0.1 with any existing GT box
3. These "extra" detections are likely true unlabeled faces
4. Add 60-80K new face annotations to the training set as pseudo-labels (s=0.3, higher noise tolerance)
5. Fine-tune from the best V8 base checkpoint for 50 epochs

**Expected gain: +0.01-0.02 mAP** (primarily helps the Hard subset).

**Cost: ~5.5h** (inference on 12,880 images + 50 epochs training).

### 3.3 Cross-Dataset Pseudo-Labeling

Add face datasets from different distributions to expand training diversity:

| Dataset | Images | Faces | Domain | Loss Weight | Expected Gain |
|---------|:------:|:-----:|--------|:-----------:|:-------------:|
| **MAFA** | 30,811 | 35,806 | Occluded faces (masks, hands, phones) | 0.5 | **+0.03-0.05** |
| **FDDB** | 2,845 | 5,171 | Unconstrained wild faces | 1.0 | +0.01 |
| **UFDD** | 6,425 | 6,425 | Adverse conditions (rain, fog, blur) | 0.3 | +0.01 |

**MAFA is the highest-value addition** because occlusion is the gimbal's real-world failure mode (hand over face, sunglasses, turning away). WIDER alone has limited occlusion coverage; MAFA directly fills this gap.

**Pseudo-label quality control:**
- MAFA/UFDD: quality threshold > 0.4 (stricter — out-of-distribution), label smoothing s=0.3
- FDDB: quality threshold > 0.35 (similar distribution to WIDER), smoothing s=0.2

**Combined expansion: 40K images → 73K total training data (5.7× base).**

### 3.4 Noisy Student Training

After pseudo-labels are generated, train a student model with **stronger augmentation** than the teacher saw:

1. **Teacher**: V8 model trained from Phase 1+2+3
2. **Inference**: Teacher generates pseudo-labels on all 73K images (WIDER test + MAFA + FDDB + UFDD + relabeled)
3. **Student**: Same architecture, trained from teacher checkpoint with:
   - RandAugment N=3, M=12 (stronger than teacher's N=2, M=9)
   - Random erasing p=0.3
   - Stochastic depth p=0.2
   - Color jitter stronger: H±50, S±50%, V±50%
4. **KL divergence loss** on pseudo-labeled data (soft targets, preserves teacher uncertainty)
5. **BCE loss** on labeled data (hard targets)
6. **Cycle**: Student becomes the new teacher → repeat (2 iterations)

**Expected gain: +0.03-0.05 mAP** over baseline pseudo-labeling.

**Cost: ~8.6h** per iteration (training with stronger augmentation is slower).

### 3.5 SCRFD Teacher Distillation

The single highest-leverage data expansion. Download the pre-trained SCRFD-34GF (9.8M params, 96.06/94.92/85.29 mAP on WIDER Face) and use it as a teacher:

```python
# One-time inference: SCRFD teacher on all training images
scrfd_teacher = onnxruntime.InferenceSession("scrfd_34g.onnx")
for img in all_training_images:  # 73K total
    dets = scrfd_teacher.run(img)
    pseudo_labels[img] = filter_quality(dets, threshold=0.4)

# Train V8 student on SCRFD teacher's labels + original GT
student = FaceFCNv8()
train(student, labeled_data + pseudo_labels)
```

**Why this works better than self-pseudo-labeling:** SCRFD-34GF has 13× more params, was NAS-optimized, and achieves 85.29 Hard mAP (vs our ~50-60 projected). Its pseudo-labels are far cleaner — especially on hard-small faces where our model is uncertain. The student learns from a much more capable teacher.

**Expected gain: +0.04-0.06 mAP** over self-pseudo-labeling alone.

**Cost: ~30 min** inference (73K images × ~15ms = ~18 min + overhead). No GPU needed for teacher (ONNX CPU inference).

### 3.6 Curriculum Threshold Scheduling

Phase in pseudo-labels by confidence rather than adding them all at once:

```
Epochs 1-5:   Only quality > 0.7  (high confidence, low noise)
Epochs 6-10:  Add quality > 0.5   (medium confidence)
Epochs 11-15: Add quality > 0.4   (full set)
```

**Why it helps:** The model first learns from the cleanest pseudo-labels, establishing solid feature baselines. Later, when lower-quality pseudo-labels are added, the model has already learned reliable detection patterns — it can use the noisy samples for fine-tuning without being destabilized.

**Gain: +0.01-0.02 mAP** over flat threshold. **Zero compute cost** (pure code change).

---

## 4. Post-Training Inference Optimizations

After the model is fully trained (all training + data expansion + distillation complete), the model is **frozen**. All further gains come from inference-side tricks with zero retraining.

### 4.1 Per-Level Threshold Calibration

Current `detect()` uses hardcoded thresholds. Instead, sweep per-level quality thresholds on the WIDER Face validation set:

| Level | Grid Cells | Density | Expected Optimal Threshold |
|-------|:----------:|:-------:|:--------------------------:|
| P3 | 19,200 | High | **0.35-0.50** (denser grid = more FP candidates, higher threshold) |
| P4 | 4,800 | Medium | **0.15-0.30** (moderate density) |
| P5 | 1,200 | Low | **0.10-0.20** (sparse grid = fewer candidates, lower threshold catches more) |

**Procedure:**
```python
for level in ["p3", "p4", "p5"]:
    for t in np.arange(0.05, 0.95, 0.05):
        preds = detect(val_set, conf_thresholds={level: t})
        map_score = compute_map(preds, val_gt)
        record(level, t, map_score)
    optimal[level] = argmax(map_score)
```

**Expected gain: +0.03-0.05 mAP.** Cost: ~1.5h inference on val set (3,226 images × 3 levels × 19 thresholds × ~30ms = ~5.5h total, but can be parallelized to ~1.5h).

### 4.2 Test-Time Flip Averaging

Run each frame twice — normal and horizontally flipped. Merge detections via NMS:

```python
def detect_tta(model, frame):
    dets1 = model.detect(frame)                    # normal
    dets2 = model.detect(frame[:, ::-1])            # flipped
    dets2 = [flip_box(d, frame.shape) for d in dets2]  # un-flip
    return soft_nms(dets1 + dets2, iou_thresh=0.5)  # merge
```

**Why it works:** False positives are rarely symmetric. A texture pattern that looks face-like in the original orientation won't match when flipped because facial features have consistent asymmetry (nose shadows, eyebrow slant). True faces (roughly symmetric) appear in both orientations. TTA filters asymmetric FPs while preserving symmetric TPs.

**Expected gain: +0.02-0.03 mAP.** Cost: 2× inference time.

### 4.3 Weighted Box Fusion (WBF)

Replace greedy NMS (even Soft-NMS) with Weighted Box Fusion. Instead of picking one box per cluster, WBF averages all overlapping boxes weighted by confidence:

```python
def wbf(boxes_list, scores_list, iou_thresh=0.5):
    # 1. Sort all detections by score descending
    # 2. For each detection, find cluster (IoU > thresh with any cluster member)
    # 3. For each cluster:
    #    - Weighted average of box coordinates (weight = score)
    #    - Final score = mean of cluster scores
    #    - Final box = Σ(score_i × box_i) / Σ(score_i)
    return fused_boxes
```

**Why WBF > NMS:** Standard NMS commits to the highest-scoring box. If the best box is slightly off-center, NMS propagates the error. WBF blends all nearby detections — if 3 boxes agree on the center but 1 is off, the weighted average centers on the majority. WIDER Face studies show +0.5-1.0 mAP gain from WBF alone.

**Expected gain: +0.01-0.02 mAP** over Soft-NMS.

### 4.4 DIoU-NMS / Adaptive NMS

Replace standard IoU in NMS with DIoU (Distance-IoU), which considers center distance:

```python
def diou(box_a, box_b):
    iou = compute_iou(box_a, box_b)
    center_dist = squared_distance(box_a.center, box_b.center)
    diagonal = squared_diagonal(smallest_enclosing_box(box_a, box_b))
    return iou - center_dist / diagonal

def diou_nms(dets, threshold=0.5):
    # Use DIoU instead of IoU for suppression
    # Two boxes that overlap loosely but have far centers are NOT suppressed
    # (they likely detect different faces)
    ...
```

**Why DIoU-NMS:** Standard IoU-NMS suppresses boxes that overlap > threshold, even if they have different centers. In crowded WIDER Face images, two nearby faces can have overlapping boxes. DIoU-NMS gives a center-distance penalty: overlapping boxes with far centers are treated as separate faces, reducing false suppression in crowd scenes.

**Adaptive NMS extension:** Adjust NMS threshold per-image based on face density. High-density images (crowds) use a lower threshold (0.3) to suppress more aggressively. Sparse images use a higher threshold (0.5) to preserve detections.

```python
face_density = estimate_face_density(image)  # heuristic from detection count
nms_thresh = 0.5 - 0.2 * face_density  # ranges 0.3-0.5
detections = nms(detections, iou_thresh=nms_thresh)
```

**Expected gain: +0.01-0.02 mAP** (DIoU-NMS: +0.01, adaptive NMS: +0.01).

### 4.5 Multi-Scale Testing

Instead of running inference at a single scale, run at 3 input scales and merge:

```python
scales = [0.9, 1.0, 1.1]  # ±10%
all_dets = []
for s in scales:
    resized = cv2.resize(frame, None, fx=s, fy=s)
    dets = detect(resized, conf_thresholds=config)
    dets = [scale_box(d, 1/s) for d in dets]  # map back to original
    all_dets.extend(dets)
final = wbf(all_dets, iou_thresh=0.5)
```

**Why it helps:** Different scales capture different face sizes. 0.9× emphasizes larger faces (better P4/P5 coverage), 1.1× captures smaller faces (better P3 coverage). The WBF merge keeps only detections that are consistent across scales — scale-specific false positives are suppressed.

**Expected gain: +0.02-0.03 mAP.** Cost: 3× inference time. For CPU deployment where frames/second is critical, skip this and use only flip TTA (2×).

### 4.6 Quality Score α/β Calibration

The quality score is currently `√(sigmoid(obj) × sigmoid(iou))`. In practice, the obj and iou branches may have different calibration. Sweep exponents:

```python
for alpha in [0.5, 1.0, 1.5, 2.0]:
    for beta in [0.5, 1.0, 1.5, 2.0]:
        quality = sigmoid(obj)^alpha × sigmoid(iou)^beta
        ap = evaluate(detections ranked by quality[alpha, beta])
        best = (alpha, beta) that maximizes ap
```

**Expected optimal:** α≈0.8-1.2, β≈1.0-1.5 (the iou branch tends to need slightly higher weight because bbox quality is more discriminative for NMS ranking than raw objectness).

**Expected gain: +0.01 mAP.** Cost: ~30 min sweep on val set.

### 4.7 Temperature Scaling

Calibrate the confidence scores by scaling logits before sigmoid:

```python
T = calibrate_temperature(val_set)  # Brent search for minimum ECE
calibrated_conf = sigmoid(logit / T)
```

T is optimized on the validation set to minimize Expected Calibration Error. Typical T=0.8-1.2. Calibrated confidences produce better precision-recall curves and hence better mAP.

**Expected gain: +0.01 mAP.** Cost: ~30 min calibration on val set.

### 4.8 FPN Cross-Level Box Voting

When the same face is detected at multiple FPN levels (P3 + P4 + P5), the three detections should agree on the box. After NMS, if boxes from different levels overlap above a relaxed threshold (IoU > 0.5), compute a weighted average:

```python
for each face cluster:
    boxes_by_level = get_boxes_by_level(cluster)
    if len(boxes_by_level) >= 2:
        weighted_avg = sum(quality_l * box_l) / sum(quality_l)
        final_box = weighted_avg  # smoother than best-level pick
```

This is especially useful for medium faces (32-64px) that straddle the P3/P4 boundary — P3 might predict slightly different coordinates than P4, and the average is more accurate than either alone.

**Expected gain: +0.01 mAP.** Cost: negligible compute (post-NMS step, ~1ms per frame).

### 4.9 Tiling for High-Resolution Inputs

If inference is ever run at resolutions above 640×480 (e.g., 1280×720 from USB camera), tile the image into overlapping crops:

- **2×2 grid** of 480×480 tiles with 48px overlap
- Run detection on each tile independently
- Merge overlapping detections via WBF (4.3)
- Effective coverage: every part of the image appears in at least one tile's center

**For 640×480 baseline:** Not needed. Reserved for high-resolution deployment scenarios.

**Expected gain: +0.01-0.02 mAP** for >640×480 inputs. Cost: 4× inference time.

### 4.10 Model Soup (Weight Averaging)

Average the weights of multiple good checkpoints from different training phases:

```python
models = [
    "v8_best.pth",               # Phase 1: peak epoch
    "v8_pseudo_best.pth",        # Phase 2: after pseudo-labeling
    "v8_mafa_best.pth",          # Phase 3: after MAFA cross-dataset
    "v8_noisy_iter2.pth",        # Phase 4: after noisy student iteration 2
]
soup_weights = average_state_dicts([m.state_dict() for m in models])
soup_model.load_state_dict(soup_weights)
```

**Why model soup works:** Each checkpoint converged to a different local minimum. The weight average lies in a flatter region of the loss landscape — the "center" of multiple minima. Flatter minima generalize better (proven in Wortsman et al., 2022).

**Expected gain: +0.02-0.03 mAP** over the best single checkpoint. Zero compute cost.

### 4.11 Detection Ensemble (Multi-Checkpoint)

Run inference with **multiple independent checkpoints** and merge their detections via WBF:

```python
checkpoints = ["v8_seed1.pth", "v8_seed2.pth", "v8_seed3.pth"]
all_dets = []
for ckpt_path in checkpoints:
    model = FaceFCNv8()
    model.load_state_dict(torch.load(ckpt_path))
    detections = model.detect(frame, **config)
    all_dets.extend(detections)
final = wbf(all_dets, iou_thresh=0.5)
```

**Why ensemble > soup:** Weight averaging blends in weight space; ensemble blends in output space. If models disagree on the same image (different false positives), the ensemble can suppress FPs that only one model sees. Ensembles benefit from **diversity** — training with different seeds, different data orderings, different SGD noise.

**Expected gain: +0.03-0.05 mAP** with 3 models. Cost: 3× inference time. Suitable for server-side deployment; not for laptop CPU real-time.

---

## 5. CPU Deployment

### 5.1 ONNX Export and INT8 Quantization

```bash
# Step 1: Export to ONNX FP32
python3 scripts/export_v8_onnx.py \
  --model models/face_cnn_v8.pth \
  --output models/face_cnn_v8.onnx \
  --input-shape 1 3 480 640 \
  --opset 17

# Step 2: Dynamic quantization to INT8
python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
  'models/face_cnn_v8.onnx',
  'models/face_cnn_v8_int8.onnx',
  weight_type=QuantType.QInt8
)
"
```

ONNX INT8 dynamic quantization converts Conv and MatMul weights to 8-bit integers while keeping activations in FP32. This gives 2-3× speedup with <1% mAP loss.

### 5.2 OpenVINO Optimization

For Intel CPU deployment, OpenVINO provides additional optimizations through graph fusion and INT8 calibration:

```bash
# Convert ONNX to OpenVINO IR
mo --input_model models/face_cnn_v8.onnx \
   --output_dir models/v8_openvino \
   --data_type FP32

# INT8 calibration with representative dataset
pot --quantize \
    --model models/v8_openvino/face_cnn_v8.xml \
    --weights models/v8_openvino/face_cnn_v8.bin \
    --output models/v8_openvino_int8 \
    --engine simplified \
    --evaluation_func eval_map
```

OpenVINO fuses BatchNorm into preceding Conv layers, merges adjacent activations, and enables INT8 inference with per-channel calibration. Expected 4-5× speedup over PyTorch FP32.

### 5.3 NCNN Deployment (Optional)

For the absolute fastest CPU inference, deploy via NCNN (Tencent's neural network inference framework optimized for mobile/edge):

```bash
# ONNX → NCNN
onnx2ncnn models/face_cnn_v8.onnx models/v8_ncnn/face_cnn_v8.param models/v8_ncnn/face_cnn_v8.bin

# INT8 quantization
ncnnint8 quantize models/v8_ncnn/face_cnn_v8.param models/v8_ncnn/face_cnn_v8.bin \
    calibration_images/ v8_int8.table

# Vulkan compute via GPU (if iGPU available)
ncnn vulkan models/v8_ncnn/face_cnn_v8.param models/v8_ncnn/face_cnn_v8.bin
```

SCRFD-0.5GF runs at 28ms on AMD Ryzen 9 3950X with NCNN INT8. Our 760K model would run at roughly 35-40ms on i7-10xxx.

### 5.4 GPU Inference via TensorRT

If the RTX 2060 is available at inference time (not just training), TensorRT provides dramatic speedups:

```bash
# FP32
trtexec --onnx=models/face_cnn_v8.onnx \
        --saveEngine=models/v8_trt_fp32.engine \
        --workspace=4096

# FP16 (2× faster than FP32 on Turing)
trtexec --onnx=models/face_cnn_v8.onnx \
        --saveEngine=models/v8_trt_fp16.engine \
        --fp16 --workspace=4096

# INT8 (requires calibration dataset)
trtexec --onnx=models/face_cnn_v8.onnx \
        --saveEngine=models/v8_trt_int8.engine \
        --int8 --calib=calibration_images/ --workspace=4096
```

TensorRT FP16 on RTX 2060 yields ~2-3ms per frame — 10,000+ face candidates/second.

### 5.5 Target Benchmarks

| Backend | Config | Expected Time | FPS | Notes |
|---------|--------|:------------:|:---:|-------|
| PyTorch FP32 | 760K params, 640×480 | ~280ms | 3.6 | Baseline, no optimization |
| ONNX FP32 | Graph opt + constant folding | ~80ms | 12.5 | 3.5× speedup over PyTorch |
| ONNX INT8 | Dynamic quantization | ~55ms | 18 | Additional 1.5× |
| OpenVINO FP32 | Model optimizer + graph fusion | ~40ms | 25 | Intel CPU optimized |
| OpenVINO INT8 | Calibrated INT8 | ~25ms | **40** | 11× speedup over baseline |
| NCNN INT8 | Mobile-optimized kernels | ~30ms | 33 | Equivalent to OpenVINO |
| TensorRT FP16 (GPU) | RTX 2060 | **~3ms** | **333** | GPU deployment |

**Target for production:** OpenVINO INT8 at ~25ms (40 FPS). This is within the 200ms gimbal tracking budget even with TTA (2× → 50ms) or multi-scale (3× → 75ms).

---

## 6. Projected Metrics and Roadmap

### 6.1 Phase Plan

| Phase | What | Time | Cumulative |
|:-----:|------|:----:|:----------:|
| **1** | **V8 implementation + training (300 ep)** | 20h | 20h |
| **2a** | Hard-aware redistribution (remaining epochs) | Included in Phase 1 | 20h |
| **2b** | WIDER re-labeling (50 ep) | 5.5h | 25.5h |
| **3a** | Baseline pseudo-labeling (3 cycles × 25 ep) | 5.2h | 30.7h |
| **3b** | Cross-dataset MAFA/FDDB/UFDD (50 ep) | 6.5h | 37.2h |
| **3c** | Curriculum threshold scheduling | 0h | 37.2h |
| **4a** | Noisy Student (2 iterations × 50 ep) | 8.6h | 45.8h |
| **4b** | SCRFD teacher distillation (1 iteration × 50 ep) | 5h | 50.8h |
| **5** | **All inference optimizations** (TTA, WBF, cal, ensemble) | 3h dev | 53.8h |
| **Total** | **Full pipeline** | **~54h** | |

**If time-constrained to 30h:** Drop Noisy Student (-8.6h) and SCRFD distillation (-5h), keep everything else. Cumulative mAP drops by ~0.05.

### 6.2 Cumulative mAP Projection

| Phase | Strategy | Stage mAP | Cumulative WIDER mAP |
|:-----:|----------|:---------:|:--------------------:|
| — | Baseline (V7 P4-only, F1=0.611) | ~0.33 | — |
| **1** | V8 architecture (3-level, independent heads, Varifocal, EIoU, DFL, BiFPN, sample-quality assignment, RandAugment, GridMask, MixUp) | **0.74** | **0.74** |
| **2a** | Hard-aware redistribution + extended 300 ep | +0.02 | 0.76 |
| **2b** | WIDER re-labeling (fix missing GT) | +0.02 | 0.78 |
| **3a** | Baseline pseudo-labeling (3-cycle) | +0.03 | 0.81 |
| **3b** | Cross-dataset MAFA/FDDB/UFDD | +0.05 | 0.86 |
| **3c** | Curriculum scheduling | +0.01 | 0.87 |
| **4a** | Noisy Student (2 iterations) | +0.03 | 0.90 |
| **4b** | SCRFD teacher distillation | +0.03 | 0.93 |
| **5a** | Threshold calibration + temperature scaling | +0.04 | 0.97 |
| **5b** | TTA flip averaging | +0.02 | 0.99 |
| **5c** | WBF + DIoU-NMS + adaptive NMS | +0.02 | 1.01 |
| **5d** | Multi-scale test (0.9×, 1.0×, 1.1×) | +0.02 | 1.03 |
| **5e** | Quality α/β + cross-level box voting | +0.02 | 1.05 |
| **5f** | Model soup (weight averaging) | +0.02 | 1.07 |
| **5g** | 3-model ensemble (WBF) | +0.03 | 1.10 |
| **Total inference stack** | | **+0.17** | |

**Notes on interpretability:** The cumulative numbers above are mAP increments summed linearly for readability. In practice, diminishing returns apply — the actual cumulative mAP is:

| Stage | WIDER Easy | WIDER Medium | WIDER Hard | **Mean mAP** |
|:-----:|:----------:|:------------:|:----------:|:------------:|
| V7 P4-only | ~0.60 | ~0.45 | ~0.12 | **~0.33** |
| V8 arch only | 89.5 | 87.0 | 60.0 | **78.8** |
| + all data expansion | 91.0 | 89.5 | 70.5 | **83.7** |
| + inference tricks | 92.0 | 90.5 | 72.0 | **84.8** |
| + soup + ensemble | 92.5 | 91.0 | 73.0 | **85.5** |

*These numbers are projections based on SCRFD-0.5GF (90.57/88.12/68.51 at 570K params) with our V8 having ~190K more backbone params, BiFPN, and all training/data/inference optimizations stacked.*

### 6.3 Success Criteria

| Threshold | mAP | Easy | Medium | Hard | CPU Time | Notes |
|:---------:|:---:|:----:|:------:|:----:|:--------:|-------|
| **Minimum** | 0.65 | 85 | 80 | 50 | <50ms | Beats SCRFD-0.5GF on Medium/Hard |
| **Target** | 0.75 | 90 | 88 | 65 | <35ms | Matches SCRFD-0.5GF with higher Hard |
| **Stretch** | 0.82 | 92 | 91 | 73 | <35ms | Exceeds SCRFD-0.5GF on all subsets |

### 6.4 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|-----------|
| P3 head still can't train (gradient cancellation) | Low (independent heads fix it mathematically) | High | Test with a 10-epoch mini-run before full training. If P3 F1 < 0.01 by ep 10, increase P3 lr 2× or reduce OHEM ratio to 3:1 |
| Hard subset mAP < 50 | Medium | Medium | Add tiling at inference (2×2, effective stride 4). Cost: 4× inference time, but run P3 at higher resolution |
| ONNX INT8 > 10% mAP loss | Medium | Medium | Fall back to ONNX FP32 (still 3× faster than PyTorch). Use OpenVINO INT8 instead (better calibration) |
| BN NaN at batch-8 | Low (V7 fix carried forward) | High | Apply running_var clamp (min=1e-4) same as V7. If persistent, switch to GroupNorm in heads (smaller models tolerate GN better, proven in SCRFD family) |
| Noisy Student destabilizes | Low | Medium | Always checkpoint before Noisy Student. If mAP drops in iteration 1, fall back to baseline pseudo-labeling only |
| SCRFD teacher inference slow | Low | Medium | Teacher runs ONNX FP32 at ~15ms per image. 73K images × 15ms = ~18 min — acceptable one-time cost |
| Training time exceeds 60h | Medium | Low | If >60h, drop Noisy Student (-8.6h), drop cross-dataset (-6.5h), keep SCRFD distillation (+5h). Total ~43h |

---

## Implementation Order

The recommended implementation order is **bottom-up**: build and validate the architecture first, then layer on training optimizations, then data expansion, then inference tricks. Each phase is independently testable on the WIDER Face val set.

```
Week 1-2:    §1 Architecture coding + §2 Training pipeline (base V8)
Week 3:      Phase 1 training (300 epochs), validate mAP
Week 4:      §3 Data expansion (pseudo-labeling, MAFA, distillation)
Week 5:      §4 Inference tricks (threshold cal, TTA, WBF, etc.)
Week 5-6:    §5 CPU deployment (ONNX, OpenVINO, benchmark)
```

---

*Document Version: 1.0 — June 1, 2026*
*Design inspired by VarifocalNet, GFL, SCRFD, EfficientDet, and lessons from FaceCNN v5-v7*
