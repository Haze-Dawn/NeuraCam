# FaceCNN v6.0 — Improvement Strategy Analysis

**Constraint:** Cannot use pretrained backbone (project requirement).
**Goal:** Improve P4 F1 ceiling, fix P2/P3 saturation, reduce false positives.

---

## Root Cause Analysis of Remaining Issues

| Issue | Root Cause | Fix Strategy |
|-------|-----------|--------------|
| P4 F1 plateaued at ~0.45 for 20 epochs | Model at local optimum, LR too low to escape | Warm restart perturbation |
| P2/P3 biases frozen at -2.38 | Progressive gating too conservative; not enough training epochs with obj loss active | Earlier gating + bias re-init |
| P2 flat saturating output (max sig 1.0, 99.8% cells >0.3) | pos_weight=50 on 10 positive cells per 76,800 — extreme class imbalance, BCE loss gradient insufficient to calibrate | Reduce pos_weight + label smoothing |
| Unknown false-positive rate | Mining disabled during training | Run fixed miner post-hoc + fine-tune |
| Head bias plateaued at -0.700 | BCEWithLogitsLoss equilibrium reached; no perturbation | Warm restart or bias re-init |

---

## Approach 1: Warm Restart from Epoch 120

**What:** Resume from epoch 120 checkpoint, reset LR to 1e-3, cosine decay for 30 additional epochs. Identical to the technique v5 used at epoch 50 (which boosted P4 F1 from 0.47 to 0.525).

**Why it would work:** The model is stuck at a local minimum — loss hasn't budged for 20 epochs, head bias frozen. A restart provides the perturbation needed to escape. The cosine schedule from 1e-3 → 1e-5 over 30 epochs gives the head enough "energy" to climb out of the local basin.

**Risk:** v5's restart destabilized all three FPN levels temporarily. P4 F1 dropped from 0.47 to 0.42 at the restart boundary, took 5 epochs to recover. v6 would see the same pattern. The EMA model should be preserved as a safety net.

**Projected impact:**
| Metric | Before (ep120) | After (+30ep) | Confidence |
|--------|:---:|:---:|:---:|
| P4 F1 | 0.451 | 0.48–0.53 | Medium |
| P3 F1 | 0.254 | 0.26–0.30 | Low-Medium |
| P2 F1 | 0.157 | 0.16–0.18 | Low |
| Head bias | -0.70 | -0.55 to -0.65 | Medium |

**Time:** ~2 hours. **Code changes:** None. **Risk:** Low.

---

## Approach 2: Hard-Negative Mining + Fine-Tuning

**What:** Run the fixed `hard_negative_mining()` (bounded heap, num_workers=0) on the best checkpoint to identify false-positive crops from WIDER train. Fine-tune the model for 15 epochs with mined hard negatives mixed into the training data (50% WIDER + 50% hard negatives).

**Why it would work:** The model trained only on WIDER Face positives and natural backgrounds. It never saw difficult false-positive patterns that look face-like. Mining identifies these explicitly. Fine-tuning on them directly reduces the false-positive rate, which should improve F1 by reducing FP counts in the F1 numerator.

**Risk:** If the model is poorly calibrated (P2 over-activation), mining will pick up noise rather than genuine hard negatives. Should mine on the best checkpoint (epoch 56, mean F1=0.319) where all three levels are balanced.

**Projected impact:**
| Metric | Before | After | Confidence |
|--------|:---:|:---:|:---:|
| P4 F1 | 0.451 | 0.46–0.50 | Medium |
| P3/P2 F1 | 0.25/0.16 | Minimal | Low (architectural limit) |
| False-positive rate | Unknown | Quantified + reduced | High |

**Time:** ~1.5 hours (20 min mining + 1-1.5 hr fine-tuning). **Code:** Already fixed. Needs `--mine-retrain` flag. **Risk:** Low.

---

## Approach 3: Earlier Progressive Gating + Bias Re-Init

**What:** Modify the v6 training script to:
- Activate P3 obj loss at epoch 3 (not 15)
- Activate P2 obj loss at epoch 5 (not 30)
- Alternatively: re-initialize P2/P3 head biases to -2.0 (not -2.5) and train with obj loss active from epoch 1
- Reduce P2 pos_weight from 50 to 25 (critical fix for saturation)

**Why it would work:** The current P2/P3 biases barely moved (-2.50 → -2.38). They had 90-105 epochs of obj loss but the signal was too weak against the initialization bias. Starting earlier with a more aggressive initial bias gives longer escape time. Reducing P2 pos_weight addresses the root mathematical cause of saturation — with 76,800 cells and only 10 positives, BCE(pos_weight=50) gives each positive 500× more gradient weight than each negative. At those ratios, the head can't calibrate.

**PosWeight analysis for P2 saturation:**
```
P2 grid: 320×240 = 76,800 cells per image
Positives: ~10 cells per image (tiny faces)
Negatives: 76,790 cells per image
BCE contribution ratio with pos_weight=50:
  Per positive cell: 50 × BCE(pred, 1.0)
  Per negative cell: 1 × BCE(pred, 0.0)
  Pos:Neg weight ratio = 50:76,790 ≈ 1:1,536
  → Negatives dominate loss 1536:1 by count, positives dominate 50:1 per-cell
  
  With pos_weight=25:
  Pos:Neg weight ratio = 25:76,790 ≈ 1:3,072
  → Better negative signal, less per-positive distortion
```

The lower pos_weight gives negatives more relative influence, which should calibrate the head away from flat-saturating output.

**Projected impact:**
| Metric | Before | After | Confidence |
|--------|:---:|:---:|:---:|
| P2 F1 | 0.157 | 0.25–0.35 | Medium-High |
| P3 F1 | 0.254 | 0.30–0.40 | Medium-High |
| P4 F1 | 0.451 | 0.47–0.52 | Medium |
| P2 saturation | 99.8% firing | Normal (~5-15% firing) | High |
| Head bias P2/P3 | -2.38 | -1.5 to -2.0 | Medium |

**Time:** Full 120-epoch retrain (~7 hours). **Code:** ~10 line changes in train_v6.py. **Risk:** Medium (retrain from scratch needed).

---

## Approach 4: Label Smoothing

**What:** Add label smoothing to BCE targets. Instead of heatmap targets being exactly 0.0 or 1.0 at each cell, smooth to 0.05/0.95 (or 0.1/0.9). This is applied during BCEWithLogitsLoss computation.

**Why it would work:** Hard 0/1 targets encourage the model to produce extreme logits — max sigmoid hitting 1.0 (which it does) and trying to push negative cells to exactly 0.0. Smoothing pulls targets inward, preventing overconfidence and improving calibration. For P2 specifically, it prevents the flat-saturating response because the model can't "win" by predicting 1.0 everywhere — the smoothed negatives at 0.05 penalize overconfident wrong predictions.

**Mathematical effect:**
```
Without smoothing: target ∈ {0, 1}
With smoothing (s=0.1): target ∈ {0.05, 0.95}
BCE loss gradient magnitude reduction near extremes:
  At pred→0: gradient reduces by ~10% (prevents logit collapse)
  At pred→1: gradient reduces by ~10% (prevents overconfident positives)
```

**Projected impact:**
| Metric | Impact | Confidence |
|--------|--------|-----------|
| P2 saturation | Likely fixed | High |
| P4 F1 | +0.02–0.05 | Medium |
| Calibration (ECE) | Improved | High |
| Training stability | Improved | High |

**Time:** Retrain needed (~7 hours). **Code:** 1 line in loss computation. **Risk:** Very low — label smoothing is standard in modern detectors.

---

## Approach 5: Detection Head Architecture Enhancement

**What:** Add a second 1×1 convolution (64 channels) before the objectness and bbox heads. Current head is a single 1×1 conv from FPN feature (64ch) directly to output (1ch obj + 4ch bbox). Adding an intermediate conv gives the head capacity to learn non-linear feature transformations before the final prediction.

**Why it would work:** The current head has exactly 65 parameters per FPN level (1×1×64→(1+4)). This is extremely lightweight — the head literally cannot represent any transformation beyond a linear projection of the FPN features. Adding an intermediate BN-ReLU-Conv block gives the head non-linear capacity. This is the RetinaNet head design.

**Cost:** +~4K params per head = +12K params total (3% increase, from 394K → 406K).

**Projected impact:**
| Metric | Impact | Confidence |
|--------|--------|-----------|
| P4 F1 | +0.03–0.08 | Medium |
| P3 F1 | +0.03–0.06 | Medium |
| P2 F1 | +0.02–0.05 | Low-Medium |
| General detection quality | Improved | Medium |

**Time:** Architecture change + retrain (~7.5 hours). **Code:** ~30 lines in face_detector_cnn.py. **Risk:** Low — standard design pattern.

---

## Approach 6: Multi-Scale Training

**What:** During training, randomly resize input between 480-800px (instead of fixed 640×480). This exposes the model to faces at different absolute scales, which particularly helps P2 (tiny face) and P3 (small face) learn scale-invariant features.

**Why it would work:** WIDER Face contains faces from 10px to full-frame. Fixed 640×480 training maps each face to a specific FPN level deterministically. Random resizing perturbs this mapping — a face that normally lands on P4 might land on P3 at a smaller scale, providing cross-level training signal. This is standard practice in all modern detectors (YOLO, RetinaNet, FCOS).

**Cost:** ~20% increase in epoch time (variable-size batches).

**Projected impact:**
| Metric | Impact | Confidence |
|--------|--------|-----------|
| P2 F1 | +0.03–0.08 | Medium |
| P3 F1 | +0.02–0.06 | Medium |
| P4 F1 | +0.01–0.03 | Low |
| Scale invariance | Improved | High |

**Time:** Retrain (~8.5 hours). **Code:** ~20 lines in dataset class. **Risk:** Low.

---

## Approach 7: EMA Ensemble Inference

**What:** Export the EMA (exponential moving average) model for inference instead of the raw model weights. The EMA model is already saved in the checkpoint (`ema_state_dict`). EMA smooths weight updates across epochs, producing a model that is more stable and often 1-3% better at inference.

**Why it would work:** The EMA model at epoch 120 has accumulated 100 epochs of weight averaging with decay=0.999. It essentially represents a weighted ensemble of the last ~1000 training steps (1/(1-0.999) ≈ 1000). This reduces variance from individual batch updates.

**Current state:** The v6 checkpoint already stores `ema_state_dict`. The export script (`onnx_export_v5.py`) needs to be pointed to use EMA weights.

**Projected impact:**
| Metric | Impact | Confidence |
|--------|--------|-----------|
| P4 F1 (inference) | +0.01–0.03 | Medium-High |
| Detection stability | Improved (less jitter) | High |

**Time:** Zero training time. Just export EMA weights. **Code:** Minor change to ONNX export. **Risk:** None.

---

## Approach 8: Post-Hoc Confidence Threshold Tuning

**What:** Run a threshold sweep on the validation set to find the optimal confidence threshold per FPN level. The current hard-coded threshold of 0.3 is arbitrary. Different FPN levels perform best at different thresholds.

**Why it would work:** P4 produces 2,641 cells >0.3. P2 produces 1.2M cells >0.3. Running a precision-recall sweep at different thresholds would identify the optimal operating point for each level independently. This is zero-cost inference optimization.

**Projected impact:**
| Metric | Impact | Confidence |
|--------|--------|-----------|
| Effective F1 | +0.02–0.05 (from threshold optimization) | High |
| False positive rate | 30-50% reduction possible | Medium |

**Time:** ~10 minutes (inference-only sweep). **Code:** Script to sweep thresholds. **Risk:** None.

---

## Approach 9: Gradient Accumulation (Larger Effective Batch)

**What:** Accumulate gradients over 2-4 batches before stepping the optimizer. With batch=16 and accumulation=2, effective batch size = 32. This is equivalent to training with batch=32 on a larger GPU.

**Why it would work:** BCEWithLogitsLoss with extreme class imbalance benefits from larger batch sizes because the positive-to-negative ratio per batch is more representative. With batch=16, some batches have 0 positive cells at P2 (extreme imbalance). With effective batch=32, the probability of zero-positive batches drops.

**Projected impact:**
| Metric | Impact | Confidence |
|--------|--------|-----------|
| P2/P3 convergence | Slightly faster | Low-Medium |
| Training stability | Improved (fewer zero-pos batches) | Medium |

**Time:** Per-epoch time increases ~10%. Total retrain ~7.7 hours. **Code:** ~5 lines in train_v6.py. **Risk:** Very low.

---

## Combined Recommendation

Given the constraints (no pretrained backbone), here's the practical path forward
in priority order:

### Phase 1: Immediate (0 training time, run right now)
1. **EMA ensemble export** — Use EMA weights for inference. Instant.
2. **Confidence threshold sweep** — Find optimal thresholds per FPN level. 10 mins.
3. **Hard-negative mining diagnostic** — Run miner on best checkpoint to quantify
   current false-positive rate. ~20 mins. No fine-tuning yet.

### Phase 2: Short (~3.5 hours total)
4. **Warm restart for 30 epochs** — Resume from ep120, LR=1e-3, cosine decay.
   Expected: P4 F1 0.45 → 0.50±0.03. ~2 hours.
5. **Hard-negative fine-tuning** — On the warm-restarted model, run miner +
   fine-tune for 15 epochs. Expected: +0.02–0.05 F1. ~1.5 hours.

### Phase 3: If results still insufficient (~8 hours)
6. **Retrain with all fixes combined:**
   - Earlier gating (P3@ep3, P2@ep5)
   - P2 pos_weight=25 (fix saturation)
   - Label smoothing (s=0.1)
   - Enhanced head (second conv layer)
   - Multi-scale training
   - Gradient accumulation (acc=2)
   
   Expected combined: P4 F1 0.52–0.58, P3 F1 0.30–0.40, P2 F1 0.25–0.35,
   P2 saturation fixed, all head biases alive.

### What will NOT work (saved for reference):
- **Reinforcement learning:** Category error. Face detection is supervised.
- **Larger backbone:** Not allowed by project constraints.
- **SGD optimizer:** AdamW with diff-LR already optimal for this setup.
- **More epochs without restart:** Model is at capacity plateau — more of the same won't help.
- **P2/P3 bias re-init alone:** Without reducing pos_weight, the mathematical saturation problem remains.
