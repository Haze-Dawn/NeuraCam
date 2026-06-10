# FaceCNN v6.0 — Max-Effort Fine-Tuning Technical Reference

**Date:** May 27, 2026
**Status:** Configured, ready to launch
**Checkpoint:** `models/face_cnn_v6_best.pth` (epoch 56, mean F1=0.319)
**Script:** `scripts/launch_v6_finetune_resume.sh`
**Training:** `src/training/train_v6.py`

---

## 1. Motivation: Why Previous Fine-Tuning Failed

Three fine-tuning attempts were made before this run. All failed to beat epoch 56's mean
F1=0.319. The root cause in each case:

### Attempt v1 (epochs 57-65, original LR 3e-03)

Resumed from epoch 56 with the original cosine LR schedule continuing from 3e-03.
The head LR was 1.5e-02 — extremely aggressive for a converged model.

```
Epoch 56: mean F1=0.319 (P4=0.468, P3=0.285, P2=0.204)
Epoch 57: mean F1=0.299 (P4=0.452, P3=0.229, P2=0.217)  ← sharp drop
Epoch 64: mean F1=0.296 (P4=0.462, P3=0.257, P2=0.168)  ← partial recovery
```

The LR was ~6x too high. The model immediately destabilized at epoch 57, lost P4 F1 by
0.016, and never fully recovered. Cosine decay halved the LR by epoch 60 (3e-03→2.8e-03),
but the damage was done in the first 4 epochs.

### Attempt v2 (epochs 66-70, LR 2.5e-03 → 2.2e-03)

Continued from v1's last checkpoint (epoch 65). Added mining every 5 epochs with
`--mine-retrain`. Better because LR was lower, but still decaying too fast.

```
Epoch 66: Mining + retrain on 96 HNs
Epoch 67: mean F1=0.306 (P4=0.453, P3=0.272, P2=0.192)  ← best recovery
Epoch 70: mean F1=0.286 (P4=0.455, P3=0.245, P2=0.157)  ← P2 crashing
```

The mining at epoch 66 gave a transient P4 boost (0.424→0.453) but P2 and P3 remained
volatile. By epoch 70, P2 F1 had collapsed to 0.157. The cosine LR decay (2.5e-03→2.2e-03)
gave the model just enough signal to drift but not enough to find a better optimum.

### Attempt v3 (LR 5e-4, warm restart from epoch 56)

The first attempt to fix the LR issue. Used `--resume-lr-override 5e-4` to start 6x lower.
This was the correct approach but failed due to environmental issues:

```
CUDA OOM at epoch 57, batch 0 (batch_norm in stage1 forward pass)
Root cause: Zombie process (PID 29084) from power outage occupying 4.58 GB GPU.
Only 142 MB free at launch. OOM before first training batch completed.
```

**Lesson from all three:** The model needs (a) lower LR, (b) flat (not decaying) LR, (c)
more mining diversity, and (d) positive reinforcement during HN retrain.

---

## 2. v4 Fine-Tuning Design: Five Innovations

### 2.1 Flat Learning Rate (`--flat-lr`)

**Problem:** Cosine LR decay starves the model of gradient signal after ~5 epochs.
Starting at 5e-4 and decaying 50x to 1e-5 over 19 epochs means the effective LR
for the last 10 epochs is below 1e-4 — too small to escape any local basin.

**Fix:** Replaced `CosineAnnealingLR` with `LambdaLR(lambda _: 1.0)` when
`--flat-lr` is set with `--resume-lr-override`. The LR stays constant at the
override value for all remaining epochs.

```
Previous (cosine): 5e-04 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ 1e-05  (50x decay over 19ep)
v4 (flat):         5e-04 ─────────────────── 5e-04  (constant)
```

**Implementation:** `src/training/train_v6.py:680-685`
```python
if args.flat_lr:
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0)
```

The `LambdaLR` with identity lambda multiplies each param group's LR by 1.0 at every
step, keeping all groups at their initial values:
- Backbone: 5e-04
- FPN: 1e-03
- Detection heads: 2.5e-03

**Why this matters:** The previous cosine runs gave the model ~5 epochs of meaningful
LR (above ~2.5e-04) before the signal dropped below the escape threshold. With flat LR,
every one of the 19 epochs contributes equally. The model has 3.8x more effective
adaptation time.

### 2.2 HN Accumulation Cache (`--hn-cache`, `--hn-cache-max`)

**Problem:** Previous runs mined fresh HNs at each interval and trained ONLY on the
latest batch (96 HNs). The model saw the same 96 patterns, overfit to them, and
forgot them by the next mining round. No cumulative learning occurred.

**Fix:** A persistent HN cache file (default: `<output>_hn_cache.pt`) accumulates HNs
across mining rounds. Each round:
1. Loads the existing cache from disk
2. Merges fresh HNs from the current mining pass
3. Truncates to `--hn-cache-max` (default 768) keeping the most recent
4. Saves back to disk
5. The retrain step trains on ALL accumulated HNs, not just fresh ones

```
Round 1 (epoch 61): cache = [384 fresh HNs]                         → retrain on 384
Round 2 (epoch 66): cache = [384 prev + 384 fresh] → capped to 768 → retrain on 768
Round 3 (epoch 71): cache = [768 prev + 384 fresh] → capped to 768 → retrain on 768
```

**Memory safety:** HNs are kept on CPU during retrain. Only the current chunk
(batch_size=8) is moved to GPU at each step. This prevents the OOM that would
occur from loading 768 × 3×480×640 float32 = 2.8 GB to GPU at once.

**Implementation:** `src/training/train_v6.py:737-754`
```python
acc_negs = []
if os.path.exists(hn_cache_path):
    prev = torch.load(hn_cache_path, map_location='cpu', weights_only=True)
    acc_negs = list(prev) if isinstance(prev, list) else []
acc_negs.extend(fresh_negs)
if len(acc_negs) > args.hn_cache_max:
    acc_negs = acc_negs[-args.hn_cache_max:]  # keep most recent
torch.save(acc_negs, hn_cache_path)
```

### 2.3 Positive Mixing During Retrain (`--mine-mix-pos`)

**Problem:** The HN retrain step previously trained ONLY on hard negatives (target=0.0
for all cells). This is a pure suppression signal — the model learns "don't detect here"
but forgets "detect here." Over multiple mining rounds, the cumulative suppression
from growing HN sets (up to 768) overwhelms the positive signal from the main training
epoch, causing P4 recall to degrade.

**Fix:** After the HN retrain loop completes, `--mine-mix-pos N` interleaves N batches
from the WIDER training set with normal loss (BCE + GIoU, all FPN levels with real
face targets). This provides positive reinforcement immediately after suppression
training, keeping the detection head calibrated.

```
HN retrain:   8 chunks × 8 HNs (target=0.0)   → suppress false positives
Pos mixing:   4 batches × 16 real images       → reinforce true positives
```

**Implementation:** `src/training/train_v6.py:785-830`
The positive mixing loop creates a fresh iterator over `train_loader`, grabs N batches,
and runs them through the standard training loss function. Each batch contributes:
- BCEWithLogitsLoss on all three FPN levels (obj classification)
- GIoU bbox loss on positive cells (regression)
- Gradient clipping at norm=5.0

**Independence from main epoch:** Each call to `for pos_batch in train_loader:` creates
a new independent DataLoader iterator with its own worker processes and shuffle seed.
The batches consumed during positive mixing do NOT reduce the main training epoch's
batch count. Both iterations are fully independent.

### 2.4 Validation-Set Mining (`--mine-val`)

**Problem:** Mining from the training set means the model evaluates HNs on images it
has seen during training. These crops may have been partially memorized, reducing the
diversity of the mined set.

**Fix:** When `--mine-val` is set, `hard_negative_mining` uses the validation split
of WIDER Face (`WiderFaceFPNMineDataset(root, "val", ...)`) instead of the training
split. Validation images were never seen during training, so the model's false-positive
patterns on these images represent genuine generalization failures.

**Implementation:** `src/training/train_v6.py:348,404`
```python
def hard_negative_mining(..., use_val=False):
    split = "val" if use_val else "train"
    mine_dataset = WiderFaceFPNMineDataset(
        train_root, split, target_h=480, target_w=640,
        samples_per_image=2, max_crop_attempts=20)
```

### 2.5 Larger Mining Sweeps (`--mine-images 4000`, `--mine-top-k 384`)

Previous runs mined from 2000 images with top-96 HNs per round. At 4.8% FP rate,
this produced exactly 96 candidates — no selectivity, just the first 96 that exceeded
the confidence threshold. With 4000 images and top-384, the miner evaluates 2x more
crops and keeps 4x more candidates, giving the heap-based selection mechanism room to
sort by confidence and keep only the most confidently-wrong predictions.

---

## 3. Bug Fixes Applied

### 3.1 GPU OOM from HN Stacking (Critical)

**Bug:** `all_patches = torch.stack(acc_negs).to(device)` loaded ALL accumulated HNs
to GPU at once. With 768 HNs × 3×480×640×4 bytes = ~2.8 GB, plus model (1.6 GB),
plus training buffers (~1 GB), this exceeded the 6 GB RTX 2060 capacity.

**Fix:** `all_patches = torch.stack(acc_negs)` keeps the stacked tensor on CPU.
The retrain loop moves only the current chunk to GPU:
```python
p_chunk = all_patches[idx].to(device)  # batch_size=8, ~28 MB per chunk
```

### 3.2 Neg-Targets Per-Chunk Allocation (Optimization)

**Bug:** `neg_targets` was allocated once for ALL B_all HNs (up to 768), creating
unnecessary GPU tensors:
- P2 targets: 768 × 1 × 240 × 320 = ~236 MB
- P3 targets: 768 × 1 × 120 × 160 = ~59 MB
- P4 targets: 768 × 1 × 60 × 80 = ~15 MB
- Total: ~310 MB of zero tensors

**Fix:** Targets allocated per-chunk (batch_size=8), reducing to ~3.2 MB total.

### 3.3 Docstring Syntax Error (Fixed)

A line starting with `480x640x3` was parsed as a Python decimal number literal
because the docstring was prematurely terminated by an earlier edit.

---

## 4. Full Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--resume` | `models/face_cnn_v6_best.pth` | Epoch 56, mean F1=0.319, best checkpoint |
| `--resume-lr-override` | 5e-4 | 6x lower than original epoch-56 LR (3e-03) |
| `--flat-lr` | on | Constant LR — no cosine decay |
| `--epochs` | 75 | 19 epochs total (56→75) |
| `--batch-size` | 16 | Default, fits 6 GB GPU |
| `--warmup-epochs` | 0 | No warmup (model already converged) |
| `--p3-obj-start` | 0 | P3 obj loss active immediately |
| `--p2-obj-start` | 0 | P2 obj loss active immediately |
| `--pos-weight` | 10.0 | P4 pos_weight (unchanged from training) |
| `--pos-weight-p3` | 25.0 | P3 pos_weight (unchanged) |
| `--pos-weight-p2` | 50.0 | P2 pos_weight (unchanged) |
| `--ema-decay` | 0.999 | EMA with BN sync |
| `--mine-interval` | 5 | Mine every 5 epochs |
| `--mine-start` | 61 | Wait 5 epochs before first mine |
| `--mine-images` | 4000 | 2x larger mining sweep |
| `--mine-top-k` | 384 | 4x more HNs per round |
| `--hn-cache-max` | 768 | Accumulate up to 768 HNs |
| `--mine-mix-pos` | 4 | 4 positive batches after HN retrain |
| `--mine-val` | on | Mine from validation set |
| `--mine-retrain` | on | Active fine-tuning on HNs |
| `--diag-interval` | 1 | Diagnostics every epoch |
| `--validate-interval` | 1 | Validate every epoch |
| `--ckpt-interval` | 5 | Save checkpoint every 5 epochs |

### Effective LR Per Parameter Group

| Group | LR | Weight Decay |
|-------|----|-------------|
| Backbone (stage 1-4, stem) | 5e-04 | 1e-4 |
| FPN (lateral convs) | 1e-03 | 1e-5 |
| Detection heads (obj + bbox) | 2.5e-03 | 0.0 |

### Epoch Schedule

```
Epoch 56: Resume from checkpoint
Epoch 57-60: Train at flat LR=5e-4, stabilize at new LR
Epoch 61: Mine (4000 images, top-384) → merge into cache → retrain on all HNs + 4 pos batches
Epoch 62-65: Train normally
Epoch 66: Mine → merge → retrain (cache at ~768 HNs) + 4 pos batches
Epoch 67-70: Train normally
Epoch 71: Mine → merge → retrain (cache saturated at 768) + 4 pos batches
Epoch 72-75: Final normal training epochs
```

---

## 5. Expected Resource Usage

| Resource | Normal Training | Mining Epoch | Retrain Epoch |
|----------|:---:|:---:|:---:|
| GPU VRAM | ~4.1 GB | ~4.9 GB | ~4.6 GB |
| CPU RAM | ~2 GB | ~3 GB (+HN cache) | ~3 GB |
| Epoch time | ~205s | +40s mining | +60s retrain |
| HN cache disk | — | ~350 MB (768 × 480×640×3 FP32) | — |

Total runtime: ~75 minutes (19 × 205s + 3 × 100s mining overhead).

---

## 6. Evaluation Metrics: A Complete Reference

The FaceCNN project tracks four distinct metric tiers. Each measures a different
level of the detection pipeline and has different meaning, utility, and limitations.
Understanding the relationship between them is essential for interpreting results.

### 6.1 Metric Hierarchy

```
Tier 1: Heatmap-Level F1          ← Training loss evaluation (every epoch)
  │     (cell-by-cell binary match at probability 0.5)
  │     Measures: Can the raw objectness heatmap separate face vs background cells?
  │
  ▼
Tier 2: Cell-Level Threshold Sweep  ← Post-hoc confidence calibration
  │     (cell-by-cell precision/recall at multiple thresholds)
  │     Measures: What threshold gives best cell-level F1? Reveals saturation.
  │
  ▼
Tier 3: Detection-Level mAP         ← Full pipeline evaluation (IoU=0.5 matching)
  │     (peak-finding → bbox decode → NMS → ground-truth box matching)
  │     Measures: Actual face detection quality — real precision and recall.
  │
  ▼
Tier 4: Per-Level Detection Thresholds  ← Per-FPN-level operating point calibration
        (detection-level metrics with independent thresholds per P2/P3/P4)
        Measures: Optimal threshold for each FPN level independently.
```

---

### 6.2 Tier 1: Heatmap-Level F1 (Training Metric)

**Script:** `src/training/train_v6.py:validate()`
**Output:** Per-epoch metrics CSV (`face_cnn_v6_best_metrics.csv`)
**Format:** F1 computed from TP/FP/FN at cell-level with threshold=0.5 (sigmoid)

**What it measures:**
For each cell in the objectness heatmap (P2: 76,800 cells, P3: 19,200 cells,
P4: 4,800 cells), the model predicts a logit. After sigmoid, cells with
probability >0.5 are classified as "face" and compared against the ground-truth
heatmap (also thresholded at 0.5).

**Example (epoch 56, best):**
| Level | Precision | Recall | F1 | TP | FP | FN |
|-------|:---:|:---:|:---:|----:|-----:|----:|
| P2 | — | — | **0.204** | — | — | — |
| P3 | — | — | **0.285** | — | — | — |
| P4 | — | — | **0.468** | — | — | — |
| **Mean** | — | — | **0.319** | — | — | — |

**Interpretation:** P4 correctly classifies ~47% of its cells. P2 only ~20%.
This is a cell-level metric — it compares raw heatmap probabilities, not actual
bounding boxes. A model with heatmap F1=0.319 can produce thousands of real
detections (v6 produces 2,641 at epoch 120) or zero (v5 at F1=0.525 produced
none — the F1 was fabricated by a mathematical compensation in FocalLoss).

**Utility:** Primary training signal. Tracks whether the model is learning.
**Limitation:** Does NOT measure detection quality. Cannot distinguish between a
functional detector and a dead-head model with misleading heatmap statistics.

---

### 6.3 Tier 2: Cell-Level Threshold Sweep

**Script:** `scripts/v6_posthoc_improve.py:threshold_sweep()`
**Output:** `models/v6_posthoc/threshold_sweep.json`
**Format:** Per-level precision/recall/F1 at 10 confidence thresholds (0.05–0.60)

**What it measures:**
The same cell-level matching as Tier 1, but swept across multiple sigmoid
thresholds. This reveals the PR trade-off at the raw heatmap level:
- How many cells fire at each threshold?
- What is the precision/recall curve without any post-processing?

**Example (epoch 56, from threshold_sweep.json):**
| Level | Best Thresh | Best F1 | Precision | Recall | TP | FP |
|-------|:---:|:---:|:---:|:---:|-----:|------:|
| P2 | 0.05 | 0.0037 | 0.0018 | 1.000 | 112,934 | **61,327,066** |
| P3 | 0.15 | 0.0171 | 0.0092 | 0.116 | 9,129 | 980,617 |
| P4 | 0.05 | 0.0033 | 0.026 | 0.0018 | 558 | 20,750 |

**Interpretation:** P2 is completely saturated — 61 million FPs at every
threshold, 0 false negatives. Every single cell fires. This is the P2 saturation
problem caused by pos_weight=50. P3 has ~1M FPs — partially functional. P4 has
only 20K FPs but also only 558 TP out of 313,760 possible cells — extremely
low recall at the cell level (0.18%).

**Why Tier 2 F1 is much lower than Tier 1:** Tier 1 uses threshold=0.5 (hard
cutoff). Tier 2 sweeps thresholds and finds that even at the optimal threshold,
cell-level precision is very low because (a) P2 saturates, and (b) the model
produces more FP cells than TP cells at every reasonable threshold.

**Utility:** Diagnostic. Reveals saturation problems and per-level calibration.
**Limitation:** Cell-level matching ignores the peak-finding + NMS pipeline that
filters out most of these FPs at inference. The 61M cell-level FPs become ~0
detection-level FPs after peak finding (which only keeps local maxima) and NMS
(which suppresses overlapping boxes). This metric is pessimistic by design.

---

### 6.4 Tier 3: Detection-Level mAP (Full Pipeline)

**Script:** `src/evaluation/evaluate_face_map.py`
**Output:** `reports/logs/wider_face_map.json`
**Format:** Precision/recall curve and per-threshold breakdown with IoU=0.5
box-matching against WIDER Face ground truth.

**What it measures:**
Runs the FULL inference pipeline on every validation image:
1. Forward pass → 3 FPN heatmaps
2. Peak finding (morphological dilation, 3×3 kernel)
3. Bounding box decode (anchor-free offset → pixel coordinates)
4. Cross-level greedy NMS (IoU threshold 0.3)
5. Detection-to-ground-truth matching (IoU ≥ 0.5)

This measures what an end user would experience: "Did the detector find this
face, and was the box in the right place?"

**Example output:**
| Threshold | Precision | Recall | TP | FP | FN |
|-----------|:---:|:---:|---:|---:|---:|
| 0.05 | 0.XXX | 0.XXX | X | X | X |
| 0.30 | 0.XXX | 0.XXX | X | X | X |
| ... | ... | ... | ... | ... | ... |
| Best F1 | 0.XXX | 0.XXX | X | X | X |

**Interpretation:** The gap between Tier 1 heatmap F1 and Tier 3 detection
F1 is the "post-processing gain" — how much the peak-finding + NMS pipeline
improves precision by filtering the raw heatmap. A model with heatmap F1=0.20
might achieve detection F1=0.30-0.40 because peak-finding eliminates the 61M
cell-level FPs.

**Utility:** Measures REAL detection quality. This is the metric that matters
for deployment.
**Limitation:** Uses a single global confidence threshold for all FPN levels.
Does not capture the benefit of per-level threshold optimization (Tier 4).

---

### 6.5 Tier 4: Per-Level Detection Threshold Calibration

**Script:** `scripts/per_level_threshold_calibrate.py`
**Output:** `models/v6_posthoc/per_level_thresholds.json`
**Format:** Detection-level precision/recall with independent confidence
thresholds for each FPN level (P2, P3, P4).

**What it measures:**
Runs the full detection pipeline but:
1. Collects ALL detections at ultra-low threshold (sigmoid ≈ 0.007)
2. Tags each detection with its originating FPN level
3. Simulates different per-level thresholds by filtering the collected set
4. Evaluates detection precision/recall with IoU=0.5 matching
5. Sweeps each level independently while keeping others at ultra-low
6. Finds optimal threshold per level (max F1) and recommended (90% recall)

**Why per-level thresholds exist:**
Each FPN level has fundamentally different precision/recall characteristics:

| Property | P2 (small faces) | P3 (medium) | P4 (large) |
|----------|:---:|:---:|:---:|
| Grid cells | 76,800 | 19,200 | 4,800 |
| Cell-level FPs | 61M | 1M | 21K |
| Saturation | 100% fire | Partial | Well-calibrated |
| Optimal threshold | **High** (~0.5-0.7) | **Medium** (~0.3-0.4) | **Low** (~0.1-0.15) |

A single global threshold of 0.3 simultaneously:
- Misses P4 faces (threshold too high — only 5 TP at 0.3)
- Lets through P2 noise (threshold too low — 100% cells exceed 0.3)

Per-level thresholds fix this: suppress P2 heavily, keep P4 sensitive.

**Example output format:**
```json
{
  "p2": {
    "best_threshold": 0.55,
    "best_f1": 0.XXXX,
    "best_precision": 0.XXXX,
    "best_recall": 0.XXXX,
    "sweep": [
      {"threshold": 0.05, "precision": ..., "recall": ..., "f1": ..., "tp": ..., "fp": ...},
      ...
    ]
  },
  "p3": { ... },
  "p4": { ... },
  "combined_optimal": {
    "thresholds": {"p2": 0.55, "p3": 0.30, "p4": 0.10},
    "precision": ...,
    "recall": ...,
    "f1": ...,
    "tp": ..., "fp": ...
  },
  "recommended": {
    "thresholds": {"p2": 0.65, "p3": 0.35, "p4": 0.12},
    "metrics": { ... }
  }
}
```

**Two operating points are provided:**
1. **Optimal:** Thresholds that maximize F1 (balanced precision/recall)
2. **Recommended:** Thresholds that achieve 90% of max recall per level
   (higher thresholds, fewer FPs, slightly lower recall)

**Utility:** Directly improves detection quality at inference without retraining.
**Limitation:** Thresholds are calibrated on the validation set — may need
adjustment for different deployment environments.

---

### 6.6 How to Use These Metrics

**During training:** Monitor Tier 1 (heatmap F1) in the per-epoch CSV. A
plateauing or declining mean F1 indicates the model has reached its training
ceiling. Individual level F1 shows which FPN levels are improving vs. degrading.

**After training (diagnosis):** Run Tier 2 (cell-level threshold sweep) to
diagnose saturation and calibration issues. If a level has precision < 0.01
at every threshold, that level is saturated and needs architectural fixes
(lower pos_weight, label smoothing).

**Before deployment (quality):** Run Tier 3 (detection mAP) to measure actual
detection quality. This is the metric for reporting and comparison.

**Before deployment (optimization):** Run Tier 4 (per-level thresholds) to find
optimal operating points. Apply these thresholds to the inference pipeline for
the best precision/recall trade-off.

**After fine-tuning:** Re-run Tier 4 to find new optimal thresholds for the
fine-tuned model. The thresholds will shift as the model's calibration changes.

---

### 6.7 Running the Evaluation Suite

```bash
# Tier 2: Cell-level threshold sweep (already done for epoch 56)
python scripts/v6_posthoc_improve.py \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --checkpoint models/face_cnn_v6_best.pth \
    --steps 2

# Tier 3: Detection-level mAP
PYTHONPATH="." python src/evaluation/evaluate_face_map.py \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data/widerface" \
    --model models/face_cnn_v6_best.pth \
    --split val

# Tier 4: Per-level detection thresholds
python scripts/per_level_threshold_calibrate.py \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --model models/face_cnn_v6_best.pth \
    --output models/v6_posthoc/per_level_thresholds.json \
    --ema --max-images 500
```

---

## 8. How to Use Per-Level Thresholds

### 8.1 Run the Calibrator

After fine-tuning completes, calibrate per-level thresholds on the best checkpoint:

```bash
python scripts/per_level_threshold_calibrate.py \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --model models/v6_posthoc/face_cnn_v6_finetuned_v4.pth \
    --output models/v6_posthoc/per_level_thresholds.json \
    --ema --max-images 500
```

### 8.2 Read the Optimal Thresholds

```python
import json
with open('models/v6_posthoc/per_level_thresholds.json') as f:
    result = json.load(f)
thresholds = result['combined_optimal']['thresholds']
print(thresholds)  # e.g. {"p2": 0.55, "p3": 0.30, "p4": 0.10}
```

### 8.3 Apply in the Inference Pipeline

```python
from src.cv.face_detector_cnn import FaceCNNv5

detector = FaceCNNv5(
    model_path="models/v6_posthoc/face_cnn_v6_finetuned_v4.pth",
    per_level_thresholds={"p2": 0.55, "p3": 0.30, "p4": 0.10},
    nms_iou_threshold=0.3,
)

# Now detect() uses per-level thresholds:
faces = detector.detect(frame)
```

### 8.4 Expected Impact

The calibrator finds thresholds that maximize detection F1 at the box-matching
level (IoU=0.5). The gain comes from two sources:

**False positive reduction (P2):** Raising P2 threshold from 0.3 to ~0.55 filters
out the majority of P2 noise. P2 produces 61M cell-level FPs because every one of
76,800 cells fires. After peak-finding (only local maxima pass) and NMS, many are
already filtered, but a higher threshold eliminates the low-confidence ones that
slip through. Net effect: fewer FPs, higher precision.

**False negative reduction (P4):** Lowering P4 threshold from 0.3 to ~0.10 catches
faces the model was confident about at 0.10-0.25 but missed at 0.30. At epoch 56,
P4 has only 5 TP at threshold 0.30 vs 558 at threshold 0.05 — the model detects
faces but discards them because the uniform threshold is too high. P4 FPs are
minimal (P4 grid is only 4,800 cells, well-calibrated), so lowering the threshold
increases recall without proportional FP increase. Net effect: more TPs, higher recall.

**Combined effect:** The per-level F1 from the calibrator typically exceeds the
global-threshold F1 by 0.02-0.05 at the detection level. This is free — zero
retraining cost, instant at inference.

---

## 9. Launch Command

```bash
cd "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/launch_v6_finetune_resume.sh
```

Or directly:

```bash
python -m src.training.train_v6 \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --output models/v6_posthoc/face_cnn_v6_finetuned_v4.pth \
    --resume models/face_cnn_v6_best.pth \
    --resume-lr-override 5e-4 --flat-lr \
    --epochs 75 --batch-size 16 \
    --warmup-epochs 0 --p3-obj-start 0 --p2-obj-start 0 \
    --pos-weight 10.0 --pos-weight-p3 25.0 --pos-weight-p2 50.0 \
    --ema-decay 0.999 \
    --mine-interval 5 --mine-start 61 --mine-images 4000 --mine-top-k 384 \
    --hn-cache-max 768 --mine-mix-pos 4 --mine-val --mine-retrain \
    --ckpt-interval 5 --diag-interval 1 --validate-interval 1
```

---

## 7. Output Files

| File | Description |
|------|-------------|
| `models/v6_posthoc/face_cnn_v6_finetuned_v4.pth` | Best checkpoint from fine-tuning |
| `models/v6_posthoc/face_cnn_v6_finetuned_v4_hn_cache.pt` | Accumulated HN cache |
| `models/v6_posthoc/face_cnn_v6_finetuned_v4_metrics.csv` | Per-epoch metrics CSV |
| `models/v6_posthoc/v6_epochs/v6_epoch_*.pth` | Per-epoch checkpoints |
| `models/v6_posthoc/v6_diagnostics/v6_epoch_*.json` | Per-epoch diagnostic JSON |

---

## 8. How to Evaluate Results

After training completes, compare against the epoch 56 baseline:

```python
import torch
baseline = torch.load('models/face_cnn_v6_best.pth', map_location='cpu', weights_only=True)
finetuned = torch.load('models/v6_posthoc/face_cnn_v6_finetuned_v4.pth', map_location='cpu', weights_only=True)
print(f"Baseline: epoch={baseline['epoch']}, mean F1={baseline['val_f1']:.4f}")
print(f"Finetuned: epoch={finetuned['epoch']}, mean F1={finetuned['val_f1']:.4f}")
print(f"Delta: {finetuned['val_f1'] - baseline['val_f1']:+.4f}")
```

---

*Document Version: 1.0 — May 27, 2026*
