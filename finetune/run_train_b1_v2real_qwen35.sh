#!/bin/bash
# Retrain 3.5-B1 on TRUE v2 data
set -e
PYTHON=/data1/env/envs/graphenhanced/bin/python3
DATA=dataset/finetune_data_v2
RUNS=runs
MODEL=/data1/models/Qwen3.5-4B
SEED=42
GPU=5

OUT=$RUNS/v2real_b1_train_a_seed${SEED}
TARGETS="q_proj k_proj v_proj o_proj in_proj_qkv out_proj"

echo "=========================================="
echo "[$(date '+%F %T')] 3.5-B1 retrain on TRUE v2 data"
echo "  data: $DATA/b1/train_a/python_java.jsonl"
echo "  out:  $OUT  gpu: $GPU"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b1 \
    --model $MODEL \
    --train_files $DATA/b1/train_a/python_java.jsonl \
    --val_files   $DATA/b1/val/python_java.jsonl \
    --output_dir  $OUT \
    --epochs 3 --lr 1e-4 --batch_size 4 --grad_accum 4 \
    --seed $SEED \
    --target_modules $TARGETS

echo "[$(date '+%F %T')] 3.5-B1 v2real done → $OUT"
