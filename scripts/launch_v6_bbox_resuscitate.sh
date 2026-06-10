#!/bin/bash
# FaceCNN v6.0 — Bbox Head Resuscitation (Targeted Fine-Tuning)
# Launched: $(date)
#
# Problem: P3/P4 bbox_pred weights collapsed to L2=0.04 (45x below init).
#   Obj heads are alive. Only bbox regression is broken.
#
# Fix:
#   1. Re-init P3/P4 bbox_pred weights + biases to kaiming init
#   2. Freeze all OTHER parameters (backbone, FPN, obj_pred, P2 bbox_pred)
#   3. Train ONLY the 512 P3/P4 bbox parameters (256 each)
#   4. 10x bbox loss weight to overcome the gradient starvation
#   5. Flat LR at 5e-3 (high — only training 512 params, no risk)
#   6. 10 epochs, no mining, no progressive gating
#
# Expected: P3/P4 bbox weights grow from L2=0.05 to L2=1.0-2.0

cd "/home/hazedawn/Documents/CV Project, Rev 3/NeuraCam Repo"

# Step 1: Create a checkpoint with re-initialized P3/P4 bbox heads
python -c "
import torch, copy
ckpt = torch.load('models/face_cnn_v6_best.pth', map_location='cpu', weights_only=True)
sd = ckpt['model_state_dict']

# Kaiming init for P3/P4 bbox_pred (fan_in mode, relu nonlinearity)
for lv in ['p3', 'p4']:
    w = sd[f'head_{lv}.bbox_pred.weight']
    b = sd[f'head_{lv}.bbox_pred.bias']
    nn = torch.nn.Conv2d(64, 4, 1)
    torch.nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
    torch.nn.init.zeros_(b)
    print(f'  Re-initialized head_{lv}.bbox_pred: L2 init={w.norm(2):.2f}')
    
torch.save(ckpt, 'models/v6_posthoc/v6_bbox_reinit_checkpoint.pth')
print('Saved: models/v6_posthoc/v6_bbox_reinit_checkpoint.pth')
"

# Step 2: Launch bbox-only training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.training.train_v6 \
    --data "/home/hazedawn/Documents/CV Project, Rev 3/Data" \
    --output models/v6_posthoc/face_cnn_v6_bbox_resuscitated.pth \
    --resume models/v6_posthoc/v6_bbox_reinit_checkpoint.pth \
    --resume-lr-override 5e-3 \
    --flat-lr \
    --epochs 10 --batch-size 16 \
    --warmup-epochs 0 \
    --p3-obj-start 0 \
    --p2-obj-start 0 \
    --pos-weight 10.0 --pos-weight-p3 25.0 --pos-weight-p2 50.0 \
    --bbox-weight 10.0 \
    --freeze-bbox-only \
    --ema-decay 0.999 \
    --mine-interval 0 \
    --ckpt-interval 2 --diag-interval 2 \
    --validate-interval 1

# Notes:
#   - freeze-bbox-only: new flag that freezes ALL params except P3/P4 bbox_pred
#   - bbox-weight 10.0: multiplies GIoU loss by 10 to overcome gradient starvation
#   - flat LR at 5e-3: aggressive but safe (only training 512 params)
#   - mine-interval 0: no mining (obj is already correct, no need for HNs)
#   - 10 epochs: should be enough to grow bbox weights from L2=0.05 to 1.0+
