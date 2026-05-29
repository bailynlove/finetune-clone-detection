#!/bin/bash
# run_train.sh — Sequential L1 training: B1 → B1-ctrl → B2 → B3
#
# All groups use unsloth/Qwen3.5-4B (same-model-different-input design).
# Runs sequentially on one GPU at a time to avoid resource contention.
# Set GPU= to the device index you want; each run occupies it fully.
#
# Per finetune_plan_v2.md §5.5 — 3 epochs, lr=1e-4, cosine+warmup, batch=4, grad_accum=4

set -e
PYTHON=/home/yanyanfu/miniconda3/envs/agentfl/bin/python3
DATA=dataset/finetune_data
RUNS=runs
MODEL=/data1/models/Qwen3.5-4B    # local path; fall back to unsloth/Qwen3.5-4B if not present
SEED=42
GPU=2          # single GPU index; change as needed

echo "=========================================="
echo "L1 training  model=$MODEL  seed=$SEED"
echo "=========================================="

# LoRA target modules for Qwen3.5 hybrid architecture (verified from source):
#   Full attention (every 4th layer):  q_proj k_proj v_proj o_proj
#   GatedDeltaNet linear attn (3/4):   in_proj_qkv out_proj
TARGETS="q_proj k_proj v_proj o_proj in_proj_qkv out_proj"

# ── B1: text-only (no image token) ────────────────────────────────────────────
echo -e "\n[$(date '+%H:%M:%S')] Starting B1 (text-only) on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b1 \
    --model $MODEL \
    --train_files $DATA/b1/train_a/python_java.jsonl \
    --val_files   $DATA/b1/val/python_java.jsonl \
    --output_dir  $RUNS/b1_train_a_seed${SEED} \
    --epochs 3 --lr 1e-4 --batch_size 4 --grad_accum 4 \
    --seed $SEED \
    --target_modules $TARGETS
echo "[$(date '+%H:%M:%S')] B1 done → $RUNS/b1_train_a_seed${SEED}"

# ── B1-ctrl: black image + text (confound control) ────────────────────────────
echo -e "\n[$(date '+%H:%M:%S')] Starting B1-ctrl (black image+text) on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b1ctrl \
    --model $MODEL \
    --train_files $DATA/b1ctrl/train_a/python_java.jsonl \
    --val_files   $DATA/b1ctrl/val/python_java.jsonl \
    --output_dir  $RUNS/b1ctrl_train_a_seed${SEED} \
    --epochs 3 --lr 1e-4 --batch_size 4 --grad_accum 4 \
    --seed $SEED \
    --target_modules $TARGETS
echo "[$(date '+%H:%M:%S')] B1-ctrl done → $RUNS/b1ctrl_train_a_seed${SEED}"

# ── B2: real image + text (main method) ───────────────────────────────────────
echo -e "\n[$(date '+%H:%M:%S')] Starting B2 (image+text) on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b2 \
    --model $MODEL \
    --train_files $DATA/b2/train_a/python_java.jsonl \
    --val_files   $DATA/b2/val/python_java.jsonl \
    --output_dir  $RUNS/b2_train_a_seed${SEED} \
    --epochs 3 --lr 1e-4 --batch_size 4 --grad_accum 4 \
    --seed $SEED \
    --target_modules $TARGETS
echo "[$(date '+%H:%M:%S')] B2 done → $RUNS/b2_train_a_seed${SEED}"

# ── B3: image-only (no code text) ─────────────────────────────────────────────
echo -e "\n[$(date '+%H:%M:%S')] Starting B3 (image-only) on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/train.py \
    --mode b3 \
    --model $MODEL \
    --train_files $DATA/b3/train_a/python_java.jsonl \
    --val_files   $DATA/b3/val/python_java.jsonl \
    --output_dir  $RUNS/b3_train_a_seed${SEED} \
    --epochs 3 --lr 1e-4 --batch_size 4 --grad_accum 4 \
    --seed $SEED \
    --target_modules $TARGETS
echo "[$(date '+%H:%M:%S')] B3 done → $RUNS/b3_train_a_seed${SEED}"

echo -e "\n=========================================="
echo "All L1 runs complete."
echo "=========================================="
