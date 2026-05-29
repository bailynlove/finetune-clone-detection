# Evaluation Results

## Dataset Notes

- **CodeNet (python-java):** Cleaned test sets — deduped against training then resampled to 1,000 pairs each (500 pos / 500 neg). B1/B2 checkpoints: `runs/b1_train_a_seed42` / `runs/b2_train_a_seed42`.
- **CodeNet (rust-java, rust-python, rust-ruby):** Original test sets (1,000 pairs each, 500 pos / 500 neg).
- **GPTCloneBench (java-python):** GPT-translated cross-language clones; 500 positive + 500 negative pairs. Single test split — SD column used, DD shows —.

SD = Same-Distribution test set (same problem pool as training, no sample overlap)  
DD = Different-Distribution test set (unseen problems)

---

## Results

| Dataset | Language Pair | Mode | Acc (SD) | Acc (DD) | Prec (SD) | Prec (DD) | Rec (SD) | Rec (DD) | F1 (SD) | F1 (DD) |
|---------|--------------|------|:--------:|:--------:|:---------:|:---------:|:--------:|:--------:|:-------:|:-------:|
| CodeNet | python-java | B1 (text FT) | 0.9970 | 0.9920 | 1.0000 | 0.9940 | 0.9940 | 0.9900 | 0.9970 | 0.9920 |
| CodeNet | python-java | B2 (img+text FT) | 0.9950 | 0.9950 | 0.9901 | 0.9920 | 1.0000 | 0.9980 | 0.9950 | 0.9950 |
| CodeNet | python-java | Zero-shot text | 0.6150 | 0.6020 | 1.0000 | 1.0000 | 0.2300 | 0.2040 | 0.3740 | 0.3389 |
| CodeNet | python-java | Zero-shot img+text | 0.6170 | 0.6020 | 1.0000 | 1.0000 | 0.2340 | 0.2040 | 0.3793 | 0.3389 |
| CodeNet | rust-java | B1 (text FT) | 0.9920 | 0.9910 | 0.9940 | 0.9881 | 0.9900 | 0.9940 | 0.9920 | 0.9910 |
| CodeNet | rust-java | B2 (img+text FT) | 0.9940 | 0.9890 | 0.9940 | 0.9822 | 0.9940 | 0.9960 | 0.9940 | 0.9891 |
| CodeNet | rust-java | Zero-shot text | 0.6060 | 0.6440 | 1.0000 | 0.9932 | 0.2120 | 0.2900 | 0.3498 | 0.4489 |
| CodeNet | rust-java | Zero-shot img+text | 0.6130 | 0.6500 | 1.0000 | 0.9934 | 0.2260 | 0.3020 | 0.3687 | 0.4632 |
| CodeNet | rust-python | B1 (text FT) | 0.9870 | 0.9890 | 0.9860 | 0.9880 | 0.9880 | 0.9900 | 0.9870 | 0.9890 |
| CodeNet | rust-python | B2 (img+text FT) | 0.9890 | 0.9890 | 0.9880 | 0.9861 | 0.9900 | 0.9920 | 0.9890 | 0.9890 |
| CodeNet | rust-python | Zero-shot text | 0.6090 | 0.6200 | 1.0000 | 1.0000 | 0.2180 | 0.2400 | 0.3580 | 0.3871 |
| CodeNet | rust-python | Zero-shot img+text | 0.6240 | 0.6370 | 1.0000 | 0.9928 | 0.2480 | 0.2760 | 0.3974 | 0.4319 |
| CodeNet | rust-ruby | B1 (text FT) | 0.9920 | 0.9910 | 0.9920 | 0.9842 | 0.9920 | 0.9980 | 0.9920 | 0.9911 |
| CodeNet | rust-ruby | B2 (img+text FT) | 0.9960 | 0.9880 | 0.9960 | 0.9822 | 0.9960 | 0.9940 | 0.9960 | 0.9881 |
| CodeNet | rust-ruby | Zero-shot text | 0.6130 | 0.6040 | 1.0000 | 0.9906 | 0.2260 | 0.2100 | 0.3687 | 0.3465 |
| CodeNet | rust-ruby | Zero-shot img+text | 0.6220 | 0.6080 | 1.0000 | 0.9909 | 0.2440 | 0.2180 | 0.3923 | 0.3574 |
| GPTCloneBench | java-python | B1 (text FT) | 1.0000 | — | 1.0000 | — | 1.0000 | — | 1.0000 | — |
| GPTCloneBench | java-python | B2 (img+text FT) | 1.0000 | — | 1.0000 | — | 1.0000 | — | 1.0000 | — |
| GPTCloneBench | java-python | Zero-shot text | 0.6530 | — | 1.0000 | — | 0.3060 | — | 0.4686 | — |
| GPTCloneBench | java-python | Zero-shot img+text | 0.7120 | — | 1.0000 | — | 0.4240 | — | 0.5955 | — |

---

## Source Files

| Row | Result JSON |
|-----|-------------|
| CodeNet python-java B1 SD | `results/b1_test_sd_1000.json` |
| CodeNet python-java B1 DD | `results/b1_test_dd_1000.json` |
| CodeNet python-java B2 SD | `results/b2_test_sd_1000.json` |
| CodeNet python-java B2 DD | `results/b2_test_dd_1000.json` |
| CodeNet python-java Zero-shot text SD | `results/zeroshot_test_sd_1000.json` |
| CodeNet python-java Zero-shot text DD | `results/zeroshot_test_dd_1000.json` |
| CodeNet python-java Zero-shot img+text SD | `results/zeroshot_b2_test_sd.json` (original test_sd) |
| CodeNet python-java Zero-shot img+text DD | `results/zeroshot_b2_test_dd.json` (original test_dd) |
| CodeNet rust-java B1 SD/DD | `results/b1_test_sd_rust_java.json` / `b1_test_dd_rust_java.json` |
| CodeNet rust-java B2 SD/DD | `results/b2_test_sd_rust_java.json` / `b2_test_dd_rust_java.json` |
| CodeNet rust-java Zero-shot text SD/DD | `results/zeroshot_b1_test_sd_rust_java.json` / `zeroshot_b1_test_dd_rust_java.json` |
| CodeNet rust-java Zero-shot img+text SD/DD | `results/zeroshot_b2_test_sd_rust_java.json` / `zeroshot_b2_test_dd_rust_java.json` |
| CodeNet rust-python B1 SD/DD | `results/b1_test_sd_rust_python.json` / `b1_test_dd_rust_python.json` |
| CodeNet rust-python B2 SD/DD | `results/b2_test_sd_rust_python.json` / `b2_test_dd_rust_python.json` |
| CodeNet rust-python Zero-shot text SD/DD | `results/zeroshot_b1_test_sd_rust_python.json` / `zeroshot_b1_test_dd_rust_python.json` |
| CodeNet rust-python Zero-shot img+text SD/DD | `results/zeroshot_b2_test_sd_rust_python.json` / `zeroshot_b2_test_dd_rust_python.json` |
| CodeNet rust-ruby B1 SD/DD | `results/b1_test_sd_rust_ruby.json` / `b1_test_dd_rust_ruby.json` |
| CodeNet rust-ruby B2 SD/DD | `results/b2_test_sd_rust_ruby.json` / `b2_test_dd_rust_ruby.json` |
| CodeNet rust-ruby Zero-shot text SD/DD | `results/zeroshot_b1_test_sd_rust_ruby.json` / `zeroshot_b1_test_dd_rust_ruby.json` |
| CodeNet rust-ruby Zero-shot img+text SD/DD | `results/zeroshot_b2_test_sd_rust_ruby.json` / `zeroshot_b2_test_dd_rust_ruby.json` |
| GPTCloneBench B1 | `results/gptclone_b1.json` |
| GPTCloneBench B2 | `results/gptclone_b2.json` |
| GPTCloneBench Zero-shot text | `results/gptclone_zeroshot.json` |
| GPTCloneBench Zero-shot img+text | `results/gptclone_zeroshot_b2.json` |
