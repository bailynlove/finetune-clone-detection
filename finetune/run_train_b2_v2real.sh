#!/bin/bash
# Retrain B2 on TRUE v2 data (with HARD_NEG)
# Output to a new dir to preserve the buggy v1-trained "v2_b2_train_a_seed42"
set -e
PYTHON=/data1/env/envs/graphenhanced/bin/python3
DATA=dataset/finetune_data_v2       # ← THE FIX: was finetune_data (v1)
RUNS=runs
MODEL=/data1/models/Qwen3.5-4B
SEED=42
GPU=6

OUT=$RUNS/v2real_b2_train_a_seed${SEED}

TARGETS="q_proj k_proj v_proj o_proj in_proj_qkv out_proj"

echo "=========================================="
echo "[$(date '+%F %T')] B2 retrain on TRUE v2 data"
echo "  data: $DATA/b2/train_a/python_java.jsonl"
echo "  out:  $OUT"
echo "  gpu:  $GPU"
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
    --target_modules $TARGETS

echo "[$(date '+%F %T')] B2 v2real done → $OUT"
