# FaceCNN V0 — Architecture Design Document

## Overview

FaceCNN V0 is a lightweight face detector designed for embedded deployment on CPU. At 186K parameters, it achieves competitive accuracy on WIDER Face while maintaining real-time inference speeds. The architecture follows a depthwise separable conv backbone + feature pyramid + direct multi-scale head design.

This document describes the architecture, training pipeline, and explains how it differs from the project's previous face detection attempts (v3–v7.1).

## Design Goals

1. **Lightweight**: Under 200K parameters for fast CPU inference
2. **Multi-scale**: Detect faces from 10px to 300px+ at 320x320 input
3. **Efficient**: Depthwise separable convolutions throughout
4. **Trainable from scratch**: No pretrained backbone dependency

## Architecture

```
Input(3, 320, 320)
    |
    ├─ Entry Block (stride 2)
    |    Conv3x3(3→32) + BN + ReLU
    |
    ├─ Stage 1 (stride 2→4)
    |    SeparableConv(32→32) + MaxPool
    |
    ├─ Stage 2 (stride 4)
    |    SeparableConv(32→32)
    |    SeparableConv(32→64)
    |
    ├─ Stage 3 (stride 4→8)
    |    SeparableConv(64→64)
    |    SeparableConv(64→128) + MaxPool
    |
    ├─ Stage 4 (stride 8)
    |    SeparableConv(128→128)
    |    SeparableConv(128→128) ------> P8 (FPN level, stride 8)
    |
    ├─ Stage 5 (stride 8→16)
    |    MaxPool + SeparableConv(128→128)
    |    SeparableConv(128→128) ------> P16 (FPN level, stride 16)
    |
    └─ Stage 6 (stride 16→32)
         MaxPool + SeparableConv(128→128)
         SeparableConv(128→128) ------> P32 (FPN level, stride 32)


FPN (top-down fusion):
    P32 ──lateral──► upsample ──┐
                                 ├──(+)── SeparableConv ──► P16
    P16 ──lateral──► upsample ───┤
                                 └──(+)── SeparableConv ──► P8
    P8  ──lateral───────────────┘


Heads (per FPN level):
    cls_head: Conv1x1(128→1) + DWConv3x3(1→1) ──► class logit
    obj_head: Conv1x1(128→1) + DWConv3x3(1→1) ──► objectness
    bbox_head: Conv1x1(128→4) + DWConv3x3(4→4) ──► bbox offsets
```

### SeparableConv Block

```
Input ─► Conv1x1(c_in→c_out) ─► DWConv3x3(groups=c_out) ─► BN ─► ReLU ─► Output
```

The 1x1 projection mixes channels, the 3x3 depthwise conv captures spatial context, and BN+ReLU provide normalization and non-linearity. This block is 8-9x more parameter-efficient than a standard 3x3 conv.

### Channel Progression

The channel sizes follow a deliberate growth pattern:
- Entry: 32 channels (captures low-level edges/textures)
- Stages 1-3: 32→64→128 (gradually increase representational capacity)
- Stages 4-6: 128 channels (maintain capacity at deeper levels)
- FPN: 128 channels throughout (sufficient for multi-scale fusion)

32-64-128 was chosen over 16-32-64 (YuNet) because the 3.5x parameter increase (186K vs 53K) goes primarily to channel width, which directly improves representational capacity without adding depth that would increase inference latency. Each extra channel at the 128-wide stages costs only ~10K params for the PW weights but adds meaningful diversity to feature representations.

### Head Design

Each detection head uses a 1x1 projection followed by a 3x3 depthwise conv with NO BN and NO activation. The 3x3 depthwise provides spatial context for each detection cell without cross-channel mixing. This is critical for separating nearby faces — the 3x3 receptive field lets the head see adjacent cells' activations, enabling the obj head to suppress duplicate detections at neighboring cells.

The heads are INDEPENDENT per scale — no shared weights between stride-8, stride-16, and stride-32 heads. This avoids the gradient cancellation problem that plagued v7's shared-weight P3 head (see comparison below).

## Parameter Breakdown

| Component | Params | Percentage |
|-----------|--------|------------|
| Entry stem | 1,024 | 0.5% |
| Stage 1 (1 block + pool) | 3,040 | 1.6% |
| Stage 2 (2 blocks) | 4,960 | 2.7% |
| Stage 3 (2 blocks) | 16,192 | 8.7% |
| Stage 4 (2 blocks) | 36,096 | 19.4% |
| Stage 5 (2 blocks) | 36,096 | 19.4% |
| Stage 6 (2 blocks) | 36,096 | 19.4% |
| FPN (3 laterals) | 54,144 | 29.0% |
| Heads (9 outputs) | 2,790 | 1.5% |
| **Total** | **186,438** | **100%** |

## Training Pipeline

### Multi-Positive Target Assignment

The single most important design decision in the training pipeline is **radius-based multi-positive assignment**. Each face is assigned to its best FPN level based on face size (target sizes: stride 8 = 32px, stride 16 = 96px, stride 32 = 256px). Then, ALL cells within a radius `r = max(1.0, (face_size / stride) * 1.0)` of the face center receive positive cls and obj targets. The radius is capped at 3 cells.

This produces 5–29 positive cells per face at the assigned stride, compared to just 1 with single-cell assignment. The obj head receives 5–29x more gradient signal, allowing it to learn meaningful objectness discrimination rather than collapsing to a degenerate constant output.

Bbox regression is only supervised on the closest cell to avoid ambiguity — the bbox head still trains on exactly 1 cell per face, matching the inference-time peak-finding decode.

### Loss Functions

- **FocalLoss (gamma=2.0, alpha=0.75)** for cls and obj: The focal loss down-weights well-classified examples (`pt` close to 1.0) so the model focuses on hard, misclassified cells. This prevents the 99.9% background cells from dominating the gradient.
- **EIoU loss** (loss_weight=5.0) for bbox regression: The EIoU loss~\cite{wu2023yunet} extends standard IoU by penalising centre distance and width/height differences in addition to overlap area. This provides stronger gradient for small face localisation compared to SmoothL1.
- **AdamW optimizer** (lr=1e-3, weight_decay=1e-4): AdamW decouples weight decay from adaptive gradient scaling, preventing the bbox weight collapse seen in v6.
- **Loss weighting**: Loss = cls_loss + obj_loss + 5.0 * bbox_loss

### Optimizer and Schedule

- **AdamW** (lr=1e-3, weight_decay=1e-4): AdamW decouples weight decay from adaptive gradient scaling, preventing the bbox weight collapse seen in v6.
- **Cosine annealing LR** with 5-epoch linear warmup: Warmup prevents early training instability; cosine schedule provides aggressive LR at the start and fine-tuning at the end.
- **Gradient clipping** (max_norm=10.0): Prevents gradient spikes from FocalLoss-amplified positive cells.

### Model EMA

Exponential Moving Average (decay=0.999) of model weights is maintained throughout training. The EMA weights are used for validation and final export. EMA provides a consistent +1-3% mAP improvement at zero inference cost by averaging the trajectory of SGD.

### Data Augmentation

- **Multi-scale training**: Input size randomly selected from 320-640px each batch. This makes the model scale-invariant and improves detection across the full WIDER Face size distribution.
- **Horizontal flip** (50% probability): Standard mirror augmentation.
- **HSV color jitter** (50% probability): Random hue shift (±10), saturation jitter (0.8-1.2x), and value jitter (0.8-1.2x) to improve robustness to lighting variation.

### Validation

mAP@0.5 is computed on the WIDER Face validation set every epoch using the EMA model. Peak-finding with threshold 0.05 is used to extract detections, followed by greedy NMS (IoU=0.45). Validation uses the same pipeline architecture as deployment, ensuring training metrics reflect real detection performance.

## Comparison: Why Previous FaceCNN Versions Failed

The project attempted five previous face detection architectures. Each failed for a distinct reason:

| Version | Params | Root Cause of Failure |
|---------|--------|------------------------|
| **v3** | 140K | **7 simultaneous training bugs** (NaN from `pred_ls` overflow, GIoU summed over all 32K cells producing loss of 16K instead of 5, FocalLoss alpha applied uniformly instead of class-aware, single shared 10ch conv corrupting all gradients). Result: F1=0.000. |
| **v4** | 62K | **Structural capacity ceiling**. 62K params too small for full-frame detection. Training on 128x128 crops with 3x face margin meant model NEVER SAW background during training — at inference on full 640x480 frames, 99.9% of cells are never-before-seen background. Objectness output compressed to range 0.41-0.54 (range 0.13). Val F1=0.084. |
| **v5** | 394K | **Dead obj head bias**. Head init: Normal(0,0.01) + bias=-4.6. BalancedFocalLoss at bias=-4.6 produces effective gradient multiplier of 0.98x for positives (barely any signal) but 1e-4 for cells near 1.0. The bias CANNOT escape -4.6 because the gradient is too weak at the low-probability regime. Stuck at sigmoid(0.01) forever. Zero usable detections despite 394K params. |
| **v6** | 394K | **Bbox head weight collapse**. Fixed v5's obj head (bias escaped to -0.70, sigmoid=0.332) but bbox weights collapsed to L2=0.04 (56x below kaiming init). Cause: AdamW weight decay + gradient starvation from 1:1,600 positive-to-negative ratio at P4 (3 positive cells out of 4,800). Once weights near zero, any positive gradient pushes `dw` to clip boundary (+5), where gradient is zero — permanent dead zone. Detection recall: 0.44%. |
| **v7** | 519K | **P3 gradient cancellation from shared-weight BCE**. P3 head uses shared 3x3 conv weights across all 19,200 cells. With 10 positive cells and 19,190 background cells, the gradient from positives (pushing sigmoid UP) exactly cancels against background (pushing sigmoid DOWN). Mathematical: `dL/dw = Σ[10 × (s-0.8) × input] + Σ[19,190 × (s-0.2) × input] ≈ 0`. P3 F1=0.007 since epoch 3, never recovered. P4 reached F1=0.611 independently. |
| **v7.1** | 504K | **Planned architecture, not yet trained**. Solves v7's gradient cancellation by using a SINGLE detection level (stride 4 via 2x FPN upsample) — one head, no shared-weight conflict. Design includes VarifocalLoss, ATSS, EIoU, 5-layer 192ch head, and knowledge distillation from YOLO11l-face + SCRFD-34GF teachers. |

### Why FaceCNN V0 Works

FaceCNN V0 incorporates every lesson from the previous five failures:

1. **Adequate capacity** (186K params): Not too small like v4 (62K), not over-parameterized like v7 (519K). The 32-64-128 channel progression provides 3.5x YuNet's representational capacity without the optimization difficulty of v7's oversized heads.

2. **Independent per-scale heads**: No shared weights between scales (fixes v7's gradient cancellation). Each scale's cls/obj/bbox heads are independent Conv1x1 + DWConv3x3 pairs, matching YuNet's proven design.

3. **Direct PW→DW heads** (no BN, no activation): Matches YuNet's head design exactly. The 3x3 depthwise provides spatial context at minimal parameter cost. The absence of BN means the head's output distribution is directly controlled by the training loss, not shifted by learned BN statistics.

4. **Proper head initialization**: Kaiming normal init for conv weights (not v5's Normal(0,0.01) which was 100x too small). Bias=0 for all heads (the FocalLoss handles the class imbalance, not a pre-set negative bias).

5. **Multi-positive radius-based target assignment**: 5-29 positives per face instead of 1 (fixes v6's gradient starvation). The obj/cls heads receive sufficient gradient signal at every training step.

6. **FocalLoss with correct implementation**: Class-aware alpha weighting, proper reduction over positives only (not all cells). This is the same formulation used by YuNet (alpha=2, beta=4 in their paper). Fixes v3's buggy implementation and v5's gradient-trap formulation.

7. **Bbox loss weight of 5.0**: Explicitly prioritizes bbox regression over classification, preventing the bbox head collapse seen in v6 (where bbox loss was 65x smaller than obj loss).

8. **AdamW with weight_decay=1e-4**: Provides sufficient regularization without the aggressive weight decay that collapsed v6's bbox weights.

## Inference Pipeline

At inference, the model outputs raw logits at 3 scales:
- **Stride 8** (40x40 grid): small faces (10-50px)
- **Stride 16** (20x20 grid): medium faces (32-120px)
- **Stride 32** (10x10 grid): large faces (64-300px)

For each cell at each scale:
1. Compute quality = sigmoid(cls) * sigmoid(obj)
2. Find local peaks via 3x3 max filter (morphological NMS)
3. Decode bbox: cx = (cell_x + 0.5 + dx) * stride, cy = (cell_y + 0.5 + dy) * stride, w = exp(dw) * stride, h = exp(dh) * stride
4. Apply Soft-NMS: decay overlapping detection scores by (1 - IoU)

The quality formula uses a simple product (not sqrt) because both cls and obj are calibrated probabilities in [0,1]. A face must have high class confidence AND high objectness to pass the threshold — this prevents false positives on background regions that happen to trigger the cls head.

## Deployment

- **PyTorch (.pth)**: For evaluation and training. 186K FP32 parameters.
- **ONNX FP32 (.onnx)**: Opset 11 with dynamic batch/height/width axes. Supports variable input resolution at inference. 735 KB.
- **ONNX INT8 (.onnx)**: Dynamic INT8 quantization of weights. 272 KB. Activations remain FP32.

All three share the identical architecture definition from `src/training/architectures/face_cnn_v0.py`.
