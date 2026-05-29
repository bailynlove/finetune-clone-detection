# Dataset

This directory holds split definitions, metadata, and build scripts.
Raw source data (Project CodeNet, XLCoST, GPTCloneBench) must be downloaded separately.
Generated images and JSONL fine-tuning files are excluded from the repository
and must be rebuilt locally using the pipeline scripts.

---

## Data Sources

### 1. Project CodeNet

**Download:**
```bash
# Mirror hosted by IBM (≈ 8 GB compressed)
wget -r -np -nH --cut-dirs=3 \
  https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
tar -xzf Project_CodeNet.tar.gz          # expands to Project_CodeNet/
```
Place the extracted directory at the repository root so the path is `./Project_CodeNet/`.

**Build splits:**
The problem-level train/val/test splits are already committed:
- `dataset/problem_splits.json` — v1 splits (all-NEG training, legacy)
- `dataset/problem_splits_v2.json` — v2 splits (HARD_NEG-aware, current)

Build the per-language-pair JSONL files:
```bash
# v1 fine-tuning data (B1 / B1ctrl / B2 / B3 modes)
python finetune/00_prepare_data.py

# v2 fine-tuning data (adds HARD_NEG same-problem AC×WA pairs)
python finetune/00_prepare_data_v2.py
```
Output lands in `dataset/finetune_data/` (v1) and `dataset/finetune_data_v2/` (v2).

---

### 2. XLCoST

**Download:**
```bash
mkdir -p XLCoST
cd XLCoST
# Official release (~285 MB)
wget https://github.com/reddy-lab-code-research/XLCoST/releases/download/v1.0/xlcost.zip
unzip xlcost.zip -d XLCoST_data/
cd ..
```

**Build eval JSONL:**
```bash
python XLCoST/build_xlcost_eval.py
```
Writes B1/B2 test JSONL files to `dataset/finetune_data/b1/xlcost_*/` and `dataset/finetune_data/b2/xlcost_*/`.

Optionally re-render XLCoST images with cleaned detokenization:
```bash
python XLCoST/rerender_xlcost.py
```

---

### 3. GPTCloneBench

**Download:**
```bash
cd GPTCloneBench
# Standalone semantic clones (≈ 30 MB)
wget https://github.com/srlabUsask/GPTCloneBench/releases/download/v1.0/GPTCloneBench_semantic_standalone_clones.zip
# Cross-language clones (≈ 6 MB total)
wget https://github.com/srlabUsask/GPTCloneBench/releases/download/v1.0/cross_language.zip
wget https://github.com/srlabUsask/GPTCloneBench/releases/download/v1.0/cross_language_part_2.zip
unzip "*.zip"
cd ..
```

**Build eval JSONL:**
```bash
python GPTCloneBench/build_java_python_eval.py        # java↔python pairs (all)
python GPTCloneBench/build_java_python_eval_2000.py   # balanced 2000-pair subset
python GPTCloneBench/build_extra_lang_pairs.py        # java↔{cpp,csharp}, python↔{cpp,csharp}
```

---

### 4. POJ-104

**Download:**
```bash
cd CodeXGLUE_POJ104
wget https://github.com/microsoft/CodeXGLUE/.../programs.tar.gz
tar -xzf programs.tar.gz
cd ..
```

**Build eval JSONL:**
```bash
python pipeline/build_poj104_eval.py
```

---

## Generating Images

After building JSONL files, generate code-snapshot PNG images:
```bash
python pipeline/02_render_images.py
```
See `dataset/images/README.md` for details on the rendering pipeline and image layout.

---

## Directory Layout (after full build)

```
dataset/
  problem_splits.json          # v1 problem-level train/val/test split
  problem_splits_v2.json       # v2 split (HARD_NEG-aware)
  dataset_summary.json         # v1 per-language-pair statistics
  dataset_summary_v2.json      # v2 per-language-pair statistics
  splits/                      # per-language-pair pair lists (v1)
  splits_v2/                   # per-language-pair pair lists (v2)
  finetune_data/               # [generated] v1 JSONL for all modes
  finetune_data_v2/            # [generated] v2 JSONL for all modes
  images/                      # [generated] 1440×896 code-snapshot PNGs
  gptclonebench_code/          # [generated] extracted GPTCloneBench source files
  poj104/                      # [generated] POJ-104 eval JSONL + images
```
