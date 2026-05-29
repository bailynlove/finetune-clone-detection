# Fine-tuning Configuration — Week 2 L1 Experiments

## Overview

Four ablation groups trained to study how different input modalities affect code clone detection accuracy.
All groups share the same base model, LoRA config, optimizer, and dataset split.

| Group | Mode | Input to model |
|-------|------|---------------|
| **B1** | `b1` | Code text only (no image) |
| **B1ctrl** | `b1ctrl` | Code text + synthetic black image (1440×896) |
| **B2** | `b2` | Code text + real rendered screenshot |
| **B3** | `b3` | Real rendered screenshot only (no code text) |

---

## Base Model

| Field | Value |
|-------|-------|
| Model | Qwen3.5-4B VLM |
| Local path | `/data1/models/Qwen3.5-4B` |
| Architecture | 32 layers: 24× GatedDeltaNet (linear attention) + 8× full attention (every 4th layer) |
| Hidden size | 2560 |
| Vision encoder | ViT depth=24, hidden=1024, patch=16×16, spatial_merge=2 |
| Vocab size | 248,320 |
| Thinking mode | Disabled (`enable_thinking=False` in chat template) |

---

## LoRA Configuration

| Hyperparameter | Value |
|----------------|-------|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `out_proj` |
| Bias | none |
| Task type | CAUSAL_LM |
| Trainable parameters | 9,830,400 (0.22% of 4.55B total) |

**Visual encoder frozen:** all parameters under `visual`, `vision_model`, `patch_embed`, `mm_projector` are frozen and wrapped in `torch.no_grad()` during forward. Gradient checkpointing is disabled on the ViT to avoid redundant recomputation.

---

## Training Hyperparameters

| Hyperparameter | B1 | B1ctrl / B2 / B3 |
|----------------|----|--------------------|
| Epochs | 3 | 3 |
| Per-device batch size | 8 | 4 |
| Gradient accumulation steps | 2 | 4 |
| **Effective batch size** | **16** | **16** |
| Learning rate | 1e-4 | 1e-4 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Precision | bf16 (4-bit NF4 base) | bf16 full precision |
| Max sequence length | 4096 | 4096 |
| Max image pixels | — | 401,408 (≈800×500) |
| Gradient checkpointing | True (non-reentrant) | True (non-reentrant) |
| Random seed | 42 | 42 |

> **Note:** B1ctrl/B2/B3 use full bf16 precision (no quantization) running in the `graphenhanced` environment (PyTorch 2.10.0+cu128). B1 was trained earlier under 4-bit NF4 quantization. The difference in batch size (8→4) for image groups is required to avoid OOM from logits allocation at long sequence lengths (~1800 tokens with images).

---

## Dataset

- **Train:** `dataset/finetune_data/{mode}/train_a/python_java.jsonl` — 20,000 pairs
- **Val:** `dataset/finetune_data/{mode}/val/python_java.jsonl` — 2,246 pairs
- **Test (same-domain):** `dataset/finetune_data/{mode}/test_sd/python_java.jsonl` — 1,000 pairs
- **Test (diff-domain):** `dataset/finetune_data/{mode}/test_dd/python_java.jsonl` — 1,000 pairs

Language pair: Python ↔ Java (cross-lingual clone detection).
Label: binary (`Yes` / `No` — is this a clone pair?).

**Loss masking:** only the final `Yes`/`No` answer token is included in the loss; the full prompt (system + code + question) is masked to -100.

---

## Image Rendering

- Resolution: 1440×896 pixels, rendered from code using the pipeline in `pipeline/`
- At `max_pixels=401408`, each image is resized to ~800×500 before ViT processing
- Each sample includes **2 images** (one per code snippet in the pair)
- ViT produces ~388 merged tokens per image after spatial merging (merge factor=2)

---

## Runtime Summary

| Group | GPU | Training time | Final train loss | Val accuracy (epoch 3) |
|-------|-----|---------------|-----------------|------------------------|
| B1 | 1× A100 80GB | 10.6 h | 0.02259 | 99.24% |
| B1ctrl | 1× A100 80GB | 7.97 h | 0.01812 | 99.51% |
| B2 | 1× A100 80GB | 8.49 h | 0.01958 | — |
| B3 | 1× A100 80GB | 4.69 h | 0.04805 | 98.09% |

---

## Script

```bash
# Example: B2
CUDA_VISIBLE_DEVICES=3 PYTORCH_ALLOC_CONF=expandable_segments:True \
python finetune/train.py \
    --mode b2 \
    --model /data1/models/Qwen3.5-4B \
    --train_files dataset/finetune_data/b2/train_a/python_java.jsonl \
    --val_files   dataset/finetune_data/b2/val/python_java.jsonl \
    --output_dir  runs/b2_train_a_seed42 \
    --epochs 3 --lr 1e-4 --batch_size 4 --grad_accum 4 --seed 42 \
    --target_modules q_proj k_proj v_proj o_proj in_proj_qkv out_proj \
    --max_pixels 401408 --no_4bit
```

Loss curve: `figures/loss_curves.png`
