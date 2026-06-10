# FaceFCN v7 — Comprehensive Benchmark Report

Generated: 2026-06-07T22:56:34.948441
Device: NVIDIA GeForce RTX 2060
Input resolution: 480×640

## Summary

| Model | mAP@0.5 | mAP(COCO) | Easy | Medium | Hard | FPS | Params |
|-------|---------|-----------|------|--------|------|-----|--------|
| v7 (P3+P4, 519K params) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0 | 519,476 |
| v7 P4-only (453K params) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.1 | 494,190 |

## Per-Size Performance

| Model | Small | Medium | Large |
|-------|-------|--------|-------|
| v7 (P3+P4, 519K params) | 0.0000 | 0.0000 | 0.0000 |
| v7 P4-only (453K params) | 0.0000 | 0.0000 | 0.0000 |

## Latency (GPU)

| Model | Mean | Median | P95 | P99 | FPS |
|-------|------|--------|-----|-----|-----|
| v7 (P3+P4, 519K params) | 1001.6ms | 982.0ms | 1289.8ms | 1432.7ms | 1.0 |
| v7 P4-only (453K params) | 76.1ms | 75.0ms | 107.1ms | 119.3ms | 13.1 |

## Confusion Matrix

| Model | TP | FP | FN | Precision | Recall |
|-------|----|----|----|-----------|--------|
| v7 (P3+P4, 519K params) | 76 | 3240783 | 39621 | 0.0000 | 0.0019 |
| v7 P4-only (453K params) | 45 | 772418 | 39652 | 0.0001 | 0.0011 |

## Top Event Categories (by mAP@0.5)

### v7 (P3+P4, 519K params)
| Category | mAP@0.5 | GT Faces |
|----------|---------|----------|
| 29--Students_Schoolkids | 0.0000 | 336 |
| 5--Car_Accident | 0.0000 | 468 |
| 28--Sports_Fan | 0.0000 | 571 |
| 10--People_Marching | 0.0000 | 2275 |
| 21--Festival | 0.0000 | 1036 |
| 0--Parade | 0.0000 | 3946 |
| 39--Ice_Skating | 0.0000 | 443 |
| 1--Handshaking | 0.0000 | 307 |
| 2--Demonstration | 0.0000 | 9105 |
| 54--Rescue | 0.0000 | 297 |

### v7 P4-only (453K params)
| Category | mAP@0.5 | GT Faces |
|----------|---------|----------|
| 28--Sports_Fan | 0.0000 | 571 |
| 5--Car_Accident | 0.0000 | 468 |
| 10--People_Marching | 0.0000 | 2275 |
| 21--Festival | 0.0000 | 1036 |
| 1--Handshaking | 0.0000 | 307 |
| 39--Ice_Skating | 0.0000 | 443 |
| 54--Rescue | 0.0000 | 297 |
| 11--Meeting | 0.0000 | 829 |
| 7--Cheering | 0.0000 | 764 |
| 18--Concerts | 0.0000 | 378 |
