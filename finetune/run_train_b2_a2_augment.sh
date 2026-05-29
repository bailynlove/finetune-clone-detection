#!/bin/bash
# A2 ablation: P0 image-augment (random theme/font/pad), 5k pairs, 1 epoch, GPU 7
# Hypothesis: multi-distribution rendering prevents layout overfitting
set -e
PYTHON=/data1/env/envs/graphenhanced/bin/python3
DATA=dataset/finetune_data_v2
MODEL=/data1/models/Qwen3.5-4B
SEED=42
GPU=7
OUT=runs/b2_a2_augment_5k

TARGETS="q_proj k_proj v_proj o_proj in_proj_qkv out_proj"

echo "=========================================="
echo "[$(date '+%F %T')] B2 A2: image-augment (dracula/github-dark/vim + fonts), 5k pairs, 1 epoch"
echo "  out: $OUT  gpu: $GPU"
echo "=========================================="

PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b2 \
    --model $MODEL \
    --train_files $DATA/b2/train_a/python_java.jsonl \
    --val_files   $DATA/b2/val/python_java.jsonl \
    --output_dir  $OUT \
    --epochs 1 --lr 1e-4 --batch_size 2 --grad_accum 8 \
    --seed $SEED \
    --limit 5000 \
    --image_augment \
    --target_modules $TARGETS

echo "[$(date '+%F %T')] A2 done → $OUT"
