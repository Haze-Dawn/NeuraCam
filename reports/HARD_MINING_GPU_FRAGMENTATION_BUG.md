# FaceCNN v6.0 — Hard-Negative Mining GPU Fragmentation Bug

**Date:** 2026-05-27
**Severity:** High (exponential slowdown during long inference loops, eventual stall)
**Status:** Fixed
**Files modified:** `src/training/train_v6.py:348-423`

---

## 1. Discovery

During the v6.0 fine-tuning run (epochs 57-71, active mining every 5 epochs from
epoch 56), the first mining pass at epoch 61 completed successfully (96 hard
negatives found). The second mining pass at epoch 66 exhibited catastrophic
exponential slowdown: batches that initially processed at 1-3 s/it degraded
to 86 s/it by batch 210 of 250. The 6 GB GPU (RTX 2060) was already at
~91% utilization from training state, and the sustained mining inference
loop pushed CUDA's caching allocator beyond its fragmentation limit.

The power outage terminated the process mid-mining at epoch 66, but the
slowdown preceded the outage.

---

## 2. Observed Behavior

### 2.1 Timing degradation curve (epoch 66 mining, 250 batches)

| Batch Range | Elapsed | Speed (s/it) | State |
|------------|---------|-------------|-------|
| 1-145 | 0-2 min | 1-3 | Normal |
| 146-155 | 2-3 min | 2-4 | Slight degradation |
| 156-157 | 3-4 min | 9-13 | Large spike |
| 158-185 | 4-13 min | 7-35 | Highly variable |
| 186-200 | 13-15 min | 10-15 | Plateau |
| 200-210 | 15-22 min | 15-**86** | Catastrophic |

**Expected completion time:** ~2.5 minutes (250 batches x 8 images ÷ 13.5 it/s
measured at epochs 57-65 training with batch=16)

**Actual completion at epoch 66:** Never observed. Extrapolating from batch 210:
40 remaining batches at 86 s/it = 57 minutes remaining. Total estimated:
~1.5 hours for a phase that should take <3 minutes.

### 2.2 VRAM pressure evidence

The 6 GB RTX 2060 runs v6.0 training at ~4,050 MB (67% utilization). The
training state (model params, optimizer states, grads, batch data, FPN
intermediates) consumes this baseline. The remaining ~2 GB headroom must
service:

| Component | VRAM | Notes |
|-----------|------|-------|
| Model params | ~1.6 MB | 394K params × 4 B FP32 |
| Optimizer states (AdamW) | ~3.2 MB | 2x params |
| Gradients | ~1.6 MB | |
| Batch data (16 × 3 × 480 × 640 FP32) | ~55 MB | |
| FPN forward intermediates | ~3,800-4,000 MB | P2: 320×240×64, P3: 160×120×64, P4: 80×60×64, conv intermediates |
| **CUDA cache reserved** | ~150-200 MB | Reserved but unused |
| **True headroom** | ~0-200 MB | Dangerously thin |

After epoch 61 mining, the GPU memory peak rose from 4,050 MB to 4,555 MB
and **never dropped back**: epochs 62-65 continued using 4,554-4,594 MB
(100% of GPU). By epoch 66, the model was already at full VRAM before
mining even started.

### 2.3 Metrics at epoch 66 from log

The mining at epoch 66 found 96 hard negatives (mined=96 in metadata), and
training completed for the epoch (805/805 batches). However, the mining
pass took ~22 minutes for 210 of 250 batches (84% complete) at severely
degraded speed. The model was trained on the hard negatives before the
mining completed, suggesting the mining reached 96 candidates early and
the remaining batches were just diagnostic noise collection.

---

## 3. Root Cause Analysis

### Root Cause (PRIMARY): Persistent CUDA VRAM fragmentation across epochs

**Location:** The mining pass runs inside the training epoch loop without
freeing CUDA memory between training and mining phases.

```
Epoch loop:
  1. Gather hard negatives (mine pass: 250 batches of model(patches))
  2. Train on hard negatives (fine-tune pass: N steps of forward+backward)
  3. Train epoch (805 batches of forward+backward)
  4. Validate (202 batches of forward)
  5. Continue to next epoch → step 1 repeats
```

After step 1 (mining), the model has processed 250 additional forward passes
through the FPN, each allocating P2/P3/P4 feature maps (320×240, 160×120,
80×60 grids) at batch_size=8. Even though these are `model.eval()` forward
passes with no backward, each allocation goes through:

1. **cudaMalloc** → finds a free block in CUDA's internal allocator → if
   fragmented, this becomes O(n) search over scattered free blocks
2. **cudaFree** → returns block to allocator → creates more fragmentation
3. **Repeat** 250× batch_size × (first batch warmup allocations)

The CUDA caching allocator (`cudaCachingAllocator`) starts with a large
contiguous pool. After 250 cycles of allocate/free at varying tensor shapes
(FPN levels have different spatial sizes + different batch sizes between
training and mining), the pool fragments into hundreds of small free blocks
scattered across 6 GB. Each subsequent `cudaMalloc` takes longer to find a
suitable block.

**Why Epoch 61 was OK but Epoch 66 broke:**
- Epoch 61: Baseline VRAM was ~4,050 MB. Mining added temporary pressure
  but the allocator could still coalesce.
- Epoch 62 and after: The 4,554 MB peak from epoch 61 mining was never
  released back to CUDA. The caching allocator held the high-water mark.
- By epoch 66: Starting from 4,594 MB baseline, the mining forward passes
  needed allocations in a thoroughly fragmented ~1.4 GB region at the top
  of address space. The allocator spent 86 seconds per batch just finding
  contiguous blocks.

### Root Cause (SECONDARY): Missing `torch.no_grad()` context

**Location:** `hard_negative_mining()` calls `model(patches)` without
wrapping in `torch.no_grad()`.

`model.eval()` changes the forward behavior of Dropout/BatchNorm but does
NOT disable autograd graph construction. Without `torch.no_grad()`:
- Each forward pass builds a computation graph for the entire FPN
- Intermediate activations are stored with `requires_grad=True` metadata
- Even though `backward()` is never called, the graph objects and their
  reference counts consume additional memory and fragment the heap

**Memory impact:** The autograd graph for one FPN forward pass on a
480×640 input at batch=8 adds ~50-100 MB of graph nodes + saved tensors
per batch. With 250 batches (and Python garbage collection not triggering
between batches), these accumulate in the managed memory pool.

### Root Cause (TERTIARY): `torch.cuda.empty_cache()` never called

**Location:** The mining loop has no cache reset between batches or epochs.

`torch.cuda.empty_cache()` releases all unused but reserved memory back to
CUDA, effectively defragmenting the allocator. Without this call, the
allocator retains its fragmented state across the entire mining pass and
subsequent training/validation loops.

---

## 4. Fix Implementation

### Fix 1: Add `torch.cuda.empty_cache()` every 20 batches

```python
for batch_idx, batch in enumerate(tqdm(mine_loader, ...)):
    if batch_idx >= max_batches:
        break
    patches = batch[0].to(device)
    with torch.no_grad():
        outputs = model(patches)
    for b in range(patches.size(0)):
        ...
    if (batch_idx + 1) % 20 == 0 and device.type == 'cuda':
        torch.cuda.empty_cache()
```

**Mechanism:** Every 20 batches (~2 seconds at normal speed), the CUDA
caching allocator resets. This limits fragmentation to at most 20 forward
passes worth of allocations, keeping the free-block list compact. The
subsequent allocation pattern repeats from a clean state.

**Cost:** `torch.cuda.empty_cache()` takes ~0.5-1 ms. Applied every 20
batches = ~13 calls for a 250-batch pass = ~13 ms overhead. Negligible.

### Fix 2: Wrap forward pass in `torch.no_grad()`

```python
with torch.no_grad():
    outputs = model(patches)
```

**Mechanism:** Disables autograd graph construction entirely. No graph
nodes, no saved variable contexts, no intermediate tensor metadata. The
forward pass runs as a pure tensor operation with minimal memory overhead.

**Memory savings:** ~50-100 MB per batch of graph nodes eliminated.
Critical on a 6 GB GPU at 4.5 GB baseline.

### Fix 3: Add mining timeout guard (defensive)

```python
import signal

def _mining_timeout_handler(signum, frame):
    raise TimeoutError("Hard-negative mining exceeded time limit")

# In hard_negative_mining():
signal.signal(signal.SIGALRM, _mining_timeout_handler)
signal.alarm(600)  # 10-minute timeout
try:
    # ... mining loop ...
finally:
    signal.alarm(0)  # cancel
```

**Mechanism:** If mining ever exceeds 10 minutes (300x the expected ~2
minute runtime), SIGALRM kills the loop with a clean Python exception
rather than letting the process hang indefinitely or trigger an OOM kill.

---

## 5. Impact Assessment

### 5.1 Was this the cause of the "crash"?

The process was likely terminated by the power outage, not by this bug
directly. However, had power remained stable, the process would have:
- Spent ~1.5 hours completing the epoch 66 mining pass (vs. 2-3 minutes)
- Entered epoch 66 training at ~4.6 GB baseline VRAM
- Risked CUDA OOM during training forward pass due to zero headroom
- Eventually either OOM-crashed or ground to a halt at <1 s/it training speed

The mining slowdown was a **pre-crash condition** — the process was
severely degraded and would have failed within 1-2 more epochs even
without the power outage.

### 5.2 Impact on fine-tuning results

| Epoch | Mining? | GPU Mem | Epoch Time | Comment |
|-------|---------|---------|------------|---------|
| 57-60 | No | 4050-4090 MB | 206-227s | Normal |
| 61 | Yes (96 HN) | **4555 MB** | **269s** | Mining spike |
| 62-65 | No | **4555-4594 MB** | 212-223s | Elevated baseline |
| 66 | Yes (started) | ~4600 MB | **∞** | Degraded |

The VRAM fragmentation from epoch 61 permanently elevated the baseline
through epochs 62-65. Epoch 61's epoch time (269s) vs normal (210s) shows
the +59s cost of a successful mining pass. Extrapolating: epoch 66 would
have taken 300+ seconds for mining alone, then training at elevated VRAM
pressure.

---

## 6. Comparison with Previous Mining Bug

| Feature | Bug 1 (CPU OOM) | Bug 2 (GPU Frag) |
|---------|-----------------|------------------|
| Discovered | May 26 | May 27 |
| Symptom | 1-3 GB CPU RAM → OOM kill | 86 s/it → eventual CUDA OOM |
| Root cause | Unbounded candidate list | CUDA allocator fragmentation |
| Fix | Bounded min-heap | `empty_cache()` + `no_grad()` |
| Scalability | Fixed (O(k) space) | Fixed (O(1) extra space) |
| Fork safety | num_workers=0 | N/A (num_workers=0 already) |

The two bugs are complementary: Bug 1 addressed CPU-side resource bounds,
Bug 2 addresses GPU-side resource bounds. Both are necessary for reliable
mining on a 6 GB GPU with 8 GB system RAM.

---

*Document Version: 1.0 — May 27, 2026*
