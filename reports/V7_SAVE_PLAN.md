# MOBILENET SAVE PLAN — FaceCNN Zero → FaceMobileNet

## The Problem
Zero's 331K from-scratch model is structurally capped at ~40% recall, limiting system F1 to ~0.60. Need F1≥0.85.

## The Fix: MobileNetV2 Backbone + Our Detection Head

| Component | Source | Value added |
|-----------|--------|-------------|
| Backbone (features) | MobileNetV2 pretrained on ImageNet | Infrastructure — we make it 2× faster with INT8 |
| Detection head (heatmap+size+offset) | Our design | Custom for gimbal, CenterNet-style |
| Training pipeline (OHEM, curriculum, etc.) | Our code | All bug fixes, engineering |
| Verifier cascade (8.9K CNN) | Our design | Novel — rejects 95% FPs |
| Temporal filtering (Kalman) | Our tracker | Gimbal-specific logic |

**Model output format similar to FaceCNN V0** — multi-scale cls/obj/bbox at strides 8/16/32.

## Architecture
```
Input 480×640
  → MobileNetV2 features[:5] (stride 8, 32ch, pretrained)
  → MobileHead (3× shared conv + 3 output heads)
  → {"heatmap": (1,60,80), "size": (2,60,80), "offset": (2,60,80)}
```

## Training Plan
| Phase | Epochs | Backbone | LR | Time |
|-------|--------|----------|-----|------|
| 1 | 0-10 | Frozen | 1e-3 (head only) | ~1h |
| 2 | 10-40 | Unfrozen | 1e-4 (all) | ~3h |
| 3 | 40-60 | Unfrozen | 1e-5 (all) | ~2h |

## Projected Metrics
| Phase | Model mAP | Model R | System F1 |
|-------|-----------|---------|-----------|
| Phase 1 (epoch 10) | 0.08-0.12 | 0.50-0.60 | 0.65-0.75 |
| Phase 2 (epoch 40) | 0.15-0.25 | 0.70-0.80 | 0.80-0.88 |
| Phase 3 (epoch 60) | 0.20-0.30 | 0.75-0.85 | **0.85-0.92** |

## Files
- `src/cv/face_mobilenet.py` — model (new)
- `src/training/train_v7.py` — training (FaceCNN V0-compatible format)
- `src/cv/face_verifier.py` — verifier (unchanged)
- `reports/V7_SAVE_PLAN.md` — this document
