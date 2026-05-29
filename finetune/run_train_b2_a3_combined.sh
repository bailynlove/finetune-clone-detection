#!/bin/bash
# A3: P0+P1 combined — image-dropout(0.3) + image-augment, 20k pairs, 3 epochs, GPU 6
set -e
PYTHON=/data1/env/envs/graphenhanced/bin/python3
DATA=dataset/finetune_data_v2
MODEL=/data1/models/Qwen3.5-4B
SEED=42
GPU=6
OUT=runs/b2_a3_combined_20k

TARGETS="q_proj k_proj v_proj o_proj in_proj_qkv out_proj"

echo "=========================================="
echo "[$(date '+%F %T')] B2 A3: image-dropout p=0.3 + image-augment, 20k pairs, 3 epochs"
echo "  out: $OUT  gpu: $GPU"
echo "=========================================="

PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b2 \
    --model $MODEL \
    --train_files $DATA/b2/train_a/python_java.jsonl \
    --val_files   $DATA/b2/val/python_java.jsonl \
    --output_dir  $OUT \
    --epochs 3 --lr 1e-4 --batch_size 2 --grad_accum 8 \
    --seed $SEED \
    --image_dropout 0.3 \
    --image_augment \
    --target_modules $TARGETS

echo "[$(date '+%F %T')] A3 done → $OUT"
