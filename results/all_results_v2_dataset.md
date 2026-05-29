# Evaluation Results — v2 Dataset (Hard Negatives)

## Dataset Design

v2 dataset fixes the label-leakage flaw in v1: in v1, `label` was 100% predictable from
`code1.problem_id == code2.problem_id`. v2 introduces hard negatives:

| Pair type | Composition | Label |
|-----------|-------------|-------|
| POS (50%) | AC(L1, P) × AC(L2, P) — same problem, both accepted | 1 |
| HARD_NEG (25%) | AC(L1, P) × WA(L2, P) — same problem, one wrong answer | 0 |
| EASY_NEG (25%) | AC(L1, P1) × AC(L2, P2), P1≠P2 — different problems | 0 |

Now same-problem pairs are only 67% positive → problem_id shortcut no longer works.

test_sd is submission-disjoint from train_a (excludes all 19,313 submission IDs in train_a).  
test_dd uses held-out problems (different-distribution).

**Checkpoint:** `runs/v2_b1_train_a_seed42` / `runs/v2_b2_train_a_seed42`  
**Language pair:** python ↔ java (train_a=20,000 pairs, test_sd=1,000, test_dd=1,000)

---

## Results — python-java

| Mode | Acc (SD) | Acc (DD) | Prec (SD) | Prec (DD) | Rec (SD) | Rec (DD) | F1 (SD) | F1 (DD) |
|------|:--------:|:--------:|:---------:|:---------:|:--------:|:--------:|:-------:|:-------:|
| B1 (text FT) | 0.8510 | 0.8250 | 0.7920 | 0.7660 | 0.9520 | 0.9360 | 0.8647 | 0.8425 |
| B2 (img+text FT) | 0.9880 | 0.9940 | 0.9980 | 0.9960 | 0.9780 | 0.9920 | 0.9879 | 0.9940 |
| Zero-shot text | 0.6010 | 0.5890 | 0.8741 | 0.8397 | 0.2360 | 0.2200 | 0.3717 | 0.3487 |
| Zero-shot img+text | 0.6040 | 0.6050 | 0.9000 | 0.8723 | 0.2340 | 0.2460 | 0.3714 | 0.3838 |

B2 FT: training completed 2026-05-18, checkpoint `runs/v2_b2_train_a_seed42`.

---

## Results — POJ-104 (OOD: C↔C same-language, trained on Python↔Java)

500 positive + 500 negative pairs from CodeXGLUE POJ-104 test set (24 problems × 500 C programs).  
All models trained on Python↔Java; POJ-104 is out-of-distribution (same language, different domain).

| Mode | Acc | Prec | Rec | F1 |
|------|:---:|:----:|:---:|:--:|
| B1 FT (v2 Python-Java ckpt) | 0.8150 | 1.0000 | 0.6300 | 0.7730 |
| B2 FT (v2 Python-Java ckpt) | 0.5060 | 1.0000 | 0.0120 | 0.0237 |
| Zero-shot text | 0.5320 | 1.0000 | 0.0640 | 0.1203 |
| Zero-shot img+text | 0.5360 | 1.0000 | 0.0720 | 0.1343 |

B2 FT POJ-104: completed 2026-05-18.  
Note: Precision=1.0 across all modes — models are extremely conservative (predict "Yes" only when certain).  
B1 FT recall=0.630 vs zero-shot recall=0.064–0.072: fine-tuning dramatically improves sensitivity even OOD.

---

## Comparison: v1 vs v2 (python-java, B1 text FT)

| Split | v1 Accuracy | v2 Accuracy | Δ |
|-------|:-----------:|:-----------:|:---:|
| test_sd | 0.9970 | 0.8510 | −0.146 |
| test_dd | 0.9920 | 0.8250 | −0.167 |

The ~15-17% drop confirms the v1 model was exploiting the problem_id shortcut.
v2 scores reflect genuine cross-language code understanding.

---

## Source Files

| Row | Result JSON |
|-----|-------------|
| v2 python-java B1 SD | `results/v2_b1_test_sd_python_java.json` |
| v2 python-java B1 DD | `results/v2_b1_test_dd_python_java.json` |
| v2 python-java Zero-shot text SD | `results/v2_zeroshot_text_test_sd_python_java.json` |
| v2 python-java Zero-shot text DD | `results/v2_zeroshot_text_test_dd_python_java.json` |
| v2 python-java Zero-shot img+text SD | `results/v2_zeroshot_b2_test_sd_python_java.json` |
| v2 python-java Zero-shot img+text DD | `results/v2_zeroshot_b2_test_dd_python_java.json` |
| POJ-104 B1 FT | `results/poj104_b1ft.json` |
| POJ-104 Zero-shot text | `results/poj104_zeroshot_text.json` |
| POJ-104 Zero-shot img+text | `results/poj104_zeroshot_imgtext.json` |
| POJ-104 B2 FT | `results/poj104_b2ft.json` |
| v2 python-java B2 SD | `results/v2_b2_test_sd_python_java.json` |
| v2 python-java B2 DD | `results/v2_b2_test_dd_python_java.json` |
