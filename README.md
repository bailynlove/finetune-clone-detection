# Cross-Language Code Clone Detection with Vision-Language Models

Fine-tuning Qwen3.5-4B on code-snapshot images + text for cross-language semantic clone detection. The project investigates how adding rendered code screenshots as a visual modality (alongside code text) affects clone detection accuracy across low-resource language pairs.

---

## Overview

### Model Modes

Four ablation groups sharing the same base model (Qwen3.5-4B VLM), LoRA config, and dataset split:

| Mode | Input | Description |
|---|---|---|
| **B1** | Text only | Code text, no image |
| **B1ctrl** | Text + black image | Synthetic 1440×896 black canvas as control |
| **B2** | Text + image | Code text + rendered code screenshot |
| **B3** | Image only | Rendered screenshot, no code text |

### Training Data Versions

| Version | NEG strategy | Notes |
|---|---|---|
| v1 | All-cross-problem NEG | Label-leakage bug: images always visually differ for NEG pairs |
| **v2** (current) | 50% POS + 25% HARD_NEG (same-problem AC×WA) + 25% EASY_NEG | Fixes leakage; forces model to distinguish semantically |

---

## Repository Structure

```
finetune/
  train.py                    # SFT trainer (B1/B1ctrl/B2/B3 modes)
  eval.py                     # Inference + F1/P/R evaluation
  eval_with_preds.py          # Per-example prediction saver (diagnosis)
  00_prepare_data.py          # Build v1 JSONL fine-tuning data
  00_prepare_data_v2.py       # Build v2 JSONL fine-tuning data
  run_train_b2_v2real.sh      # B2 v2real training (GPU 6)
  run_train_b1_v2real_qwen35.sh  # B1 v2real training, Qwen3.5 (GPU 5)
  run_train_b1_v2real_coder.sh   # B1 v2real training, Qwen2.5-Coder (GPU 7)
  run_train_b2_a1_dropout.sh  # Ablation A1: image-dropout p=0.3
  run_train_b2_a2_augment.sh  # Ablation A2: on-the-fly image augmentation
  run_train_b2_a3_combined.sh # Ablation A3: A1+A2 combined

pipeline/
  01_build_dataset_v2.py      # Problem-level split generation (v2)
  02_render_images.py         # Render code → 1440×896 PNG
  03_dataset_figures.py       # Dataset analysis figures

XLCoST/
  build_xlcost_eval.py        # Build XLCoST B1/B2 eval JSONL
  rerender_xlcost.py          # Re-render XLCoST images with cleaned code

GPTCloneBench/
  build_java_python_eval.py   # GPTCloneBench eval JSONL
  build_extra_lang_pairs.py   # Additional language pair evals

dataset/
  problem_splits_v2.json      # Problem-level train/val/test split (v2)
  splits_v2/                  # Per-language-pair pair lists (v2)
  README.md                   # Download and build instructions
  images/README.md            # Image rendering details

results/
  v2real/                     # v2real model evaluation JSON files
  ablations/                  # P0/P1 ablation evaluation JSON files

docs/
  finetune_config.md          # Model/LoRA/optimizer reference
  b2_v2real_xlcost_diagnosis.md  # B2 OOD failure analysis and fix proposals
```

---

## Environment

```bash
conda create -n clone python=3.10
pip install torch transformers peft trl datasets pillow pygments
# Flash-attention (optional, speeds up training)
pip install flash-attn --no-build-isolation
```

Model weights: `Qwen3.5-4B` from Hugging Face or a local path.

---

## Quick Start

### Training

```bash
# B1 (text-only), v2 data, Qwen3.5-4B
CUDA_VISIBLE_DEVICES=0 python finetune/train.py \
    --mode b1 --model /path/to/Qwen3.5-4B \
    --train_files dataset/finetune_data_v2/b1/train_a/python_java.jsonl \
    --val_files   dataset/finetune_data_v2/b1/val/python_java.jsonl \
    --output_dir  runs/my_b1 \
    --epochs 3 --lr 1e-4 --batch_size 2 --grad_accum 8 --seed 42 \
    --target_modules q_proj k_proj v_proj o_proj in_proj_qkv out_proj

# B2 (text + image) with OOD robustness fixes (recommended)
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python finetune/train.py \
    --mode b2 --model /path/to/Qwen3.5-4B \
    --train_files dataset/finetune_data_v2/b2/train_a/python_java.jsonl \
    --val_files   dataset/finetune_data_v2/b2/val/python_java.jsonl \
    --output_dir  runs/my_b2 \
    --epochs 1 --lr 1e-4 --batch_size 2 --grad_accum 8 --seed 42 \
    --image_dropout 0.3 --image_augment \
    --target_modules q_proj k_proj v_proj o_proj in_proj_qkv out_proj
```

Key training flags:
- `--image_dropout P` — replace a pair's images with blank background at probability P (P1 fix)
- `--image_augment` — re-render each batch with a random dark theme / font / padding (P0 fix)
- `--limit N` — truncate training data to first N records (ablations)

### Evaluation

```bash
# Zero-shot (no checkpoint)
python finetune/eval.py --mode b2 --no_lora \
    --base_model /path/to/Qwen3.5-4B \
    --test_files dataset/finetune_data/b2/xlcost_java_python/test.jsonl \
    --output results/my_zs_b2_xlcost.json

# Fine-tuned
python finetune/eval.py --mode b2 --checkpoint runs/my_b2 \
    --base_model /path/to/Qwen3.5-4B \
    --test_files dataset/finetune_data_v2/b2/test_sd/python_java.jsonl \
    --output results/my_b2_codenet_sd.json
```

---

## Results

All experiments use **Qwen3.5-4B** as the base model unless noted.
Training data: `train_a` split, Python↔Java pairs only.
`test_sd` = submission-disjoint from train; `test_dd` = problem-disjoint.

### B1 (Text-Only) — XLCoST Cross-Language F1

| Model | j↔py | j↔cpp | j↔cs | cpp↔py | cpp↔cs | py↔cs | **Avg** |
|---|---|---|---|---|---|---|---|
| 3.5 ZS | 0.768 | 0.661 | 0.802 | 0.814 | 0.789 | 0.739 | 0.762 |
| 3.5 B1-old (v1) | 0.921 | 0.919 | 0.896 | 0.943 | 0.928 | 0.918 | 0.921 |
| **3.5 B1-v2real** | **0.950** | **0.933** | **0.926** | **0.946** | **0.936** | **0.936** | **0.938** |
| Coder-3B ZS | 0.996 | 1.000 | 1.000 | 0.996 | 0.988 | 0.996 | 0.996 |
| Coder-3B B1-old | 0.913 | 0.914 | 0.941 | 0.904 | 0.947 | 0.878 | 0.916 |
| **Coder-3B B1-v2real** | **0.935** | **0.944** | **0.929** | **0.937** | **0.934** | **0.910** | **0.932** |

> Coder-3B = Qwen2.5-Coder-3B-Instruct (B1 mode only). Its ZS score on XLCoST (0.996) is anomalously high — the model happens to perform well on GeeksforGeeks-style code zero-shot, but its CodeNet score is poor (0.556/0.621), indicating prompt-style sensitivity.

### B1 (Text-Only) — CodeNet F1

| Model | test_sd | test_dd |
|---|---|---|
| 3.5 ZS | 0.372 | 0.349 |
| 3.5 B1-old | 0.865 | 0.842 |
| **3.5 B1-v2real** | **0.866** | **0.846** |
| Coder-3B ZS | 0.556 | 0.621 |
| Coder-3B B1-old | 0.786 | 0.759 |
| **Coder-3B B1-v2real** | **0.813** | **0.756** |

### B2 (Text + Image) — XLCoST Cross-Language F1

| Model | j↔py | j↔cpp | j↔cs | cpp↔py | cpp↔cs | py↔cs | **Avg** |
|---|---|---|---|---|---|---|---|
| 3.5 ZS-B2 | 0.745 | 0.786 | 0.797 | 0.792 | 0.771 | 0.753 | 0.774 |
| 3.5 B2-old (v1, bug) | 0.000 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.001 |
| 3.5 B2-v2real | 0.512 | 0.565 | 0.553 | 0.466 | 0.586 | 0.527 | 0.535 |
| **A1: dropout 5k/1ep** | **0.986** | **0.984** | **0.978** | **0.980** | **0.984** | **0.989** | **0.983** |
| **A2: augment 5k/1ep** | **0.985** | **0.985** | **0.980** | **0.984** | **0.984** | **0.992** | **0.985** |
| A3: combined 20k/3ep | 0.747 | 0.667 | 0.693 | 0.710 | 0.703 | 0.758 | 0.713 |

### B2 (Text + Image) — CodeNet F1

| Model | test_sd | test_dd |
|---|---|---|
| 3.5 ZS-B2 | 0.376 | 0.372 |
| 3.5 B2-old (v1, bug) | 0.687 | 0.764 |
| 3.5 B2-v2real | 0.868 | 0.867 |
| A1: dropout 5k/1ep | 0.831 | — |
| A2: augment 5k/1ep | 0.842 | — |
| A3: combined 20k/3ep | 0.857 | 0.836 |

### GPTCloneBench — Java↔Python (2000-pair balanced subset)

Semantic cross-language clones from GPTCloneBench. All pairs are Java↔Python.
"B1-old / B2-old" = v1 training data; "B1-v2 / B2-v2" = v2 training data (HARD_NEG).

| Model | F1 | Precision | Recall |
|---|---|---|---|
| 3.5 ZS-B1 | 0.486 | 1.000 | 0.321 |
| 3.5 ZS-B2 | 0.596 | 1.000 | 0.425 |
| Coder-3B ZS-B1 | 0.833 | 1.000 | 0.713 |
| Coder-3B ZS-B2 | 0.832 | 1.000 | 0.712 |
| Coder-3B B1-old | 0.672 | 1.000 | 0.506 |
| 3.5 B1-old | 0.998 | 0.999 | 0.997 |
| **3.5 B2-old** | **0.999** | 0.999 | 0.999 |
| 3.5 B1-v2 | 0.790 | 0.998 | 0.653 |
| **3.5 B2-v2** | **0.970** | 0.999 | 0.943 |

> Note: GPTCloneBench is same-language (Java↔Python only) and in-distribution for models trained on Python↔Java pairs, so high scores for B1-old/B2-old are expected. The v2 models score lower because HARD_NEG training makes the model more conservative.

### GPTCloneBench — Cross-Language (v2 models, F1)

Four language pairs from GPTCloneBench cross-language evaluation.

| Model | j↔cpp | j↔cs | py↔cpp | py↔cs | **Avg** |
|---|---|---|---|---|---|
| 3.5 ZS-text | 0.301 | 0.589 | 0.461 | 0.651 | 0.500 |
| 3.5 ZS-B2 | 0.398 | 0.698 | 0.572 | 0.737 | 0.601 |
| Coder-3B ZS-text | 0.647 | 0.962 | 0.556 | 0.919 | 0.771 |
| Coder-3B ZS-B2 | 0.626 | 0.920 | 0.637 | 0.875 | 0.765 |
| Coder-3B B1-old | 0.626 | 0.806 | 0.552 | 0.716 | 0.675 |
| 3.5 B1-v2 | 0.777 | 0.860 | 0.859 | 0.874 | 0.843 |
| **3.5 B2-v2** | **0.971** | **0.996** | **0.986** | **0.997** | **0.988** |

---

## Key Findings

### 1. v1 → v2 Training Data Fix

B2-old was trained on v1 data where all NEG pairs come from different problems, making images trivially distinguishable by visual layout. This caused complete OOD failure (XLCoST F1 = 0.001). v2 introduces HARD_NEG (same-problem AC×WA pairs) to break this visual shortcut.

### 2. B2-v2real OOD Failure: Visual Modal Collapse

Even after fixing the data bug, B2-v2real underperforms zero-shot B2 on XLCoST (0.535 vs 0.774). Diagnosis in `docs/b2_v2real_xlcost_diagnosis.md`:

- All XLCoST failures are **recall-side only** (Precision = 1.000 for all models)
- B2-v2real recall: 0.901 (in-distribution val) → **0.366** (OOD XLCoST)
- Per-example analysis: B1-v2real recovers 84% of B2-v2real's errors using identical text input — the only difference is images
- Root cause: LoRA adapts the LLM to weight visual embeddings heavily for the CodeNet rendering distribution; OOD images (XLCoST uses GFG-style code with `class GFG{}` wrappers and shorter snippets) trigger conservative rejection

### 3. OOD Fixes: P0 (Image Augmentation) and P1 (Image Dropout)

Two training-time interventions that break visual over-reliance:

**P1 (image-dropout):** Replace both images in a pair with a blank background at probability p=0.3, forcing the model to learn text-based fallback paths.

**P0 (image-augment):** Re-render code from text on every batch with randomly chosen dark theme (monokai / dracula / github-dark / vim / native), font (DejaVu / Liberation Mono), and padding.

Results at 5k pairs / 1 epoch: XLCoST F1 **0.983** (A1) and **0.985** (A2), far exceeding both ZS-B2 (0.774) and B1-v2real (0.938).

### 4. Training Duration Matters

Full 20k × 3 epochs with P0+P1 (A3) gives XLCoST = 0.713 — worse than the 5k/1ep ablations. Per-epoch XLCoST degrades monotonically (epoch 1 → 2 → 3) even as val loss improves. This confirms the model progressively re-learns CodeNet visual shortcuts despite augmentation. The optimal regime appears to be **20k × 1 epoch** (pending).

---

## Citation

If you use this code or data, please cite the relevant base datasets:
- Project CodeNet: Puri et al., 2021
- XLCoST: Zhu et al., 2022
- GPTCloneBench: Alam et al., 2023
