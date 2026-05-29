# Complete Evaluation Results

Train data: python_java only. rust_* pairs are zero-shot cross-lingual generalization.

## python_java  (训练内)

| Model                  | Split    |    Acc |   Prec |    Rec |     F1 |
|------------------------|----------|--------|--------|--------|--------|
| B1 (text FT)           | test_sd  | 0.9950 | 0.9980 | 0.9920 | 0.9950 |
| B1ctrl (blk+text FT)   | test_sd  | 0.9950 | 0.9940 | 0.9960 | 0.9950 |
| B2 (img+text FT)       | test_sd  | 0.9980 | 0.9960 | 1.0000 | 0.9980 |
| B3 (img-only FT)       | test_sd  | 0.9890 | 0.9939 | 0.9840 | 0.9889 |
| Zero-shot text         | test_sd  | 0.6120 | 1.0000 | 0.2240 | 0.3660 |
| Zero-shot img+text     | test_sd  | 0.6170 | 1.0000 | 0.2340 | 0.3793 |
| B1 (text FT)           | test_dd  | 0.9920 | 0.9940 | 0.9900 | 0.9920 |
| B1ctrl (blk+text FT)   | test_dd  | 0.9920 | 0.9920 | 0.9920 | 0.9920 |
| B2 (img+text FT)       | test_dd  | 0.9940 | 0.9901 | 0.9980 | 0.9940 |
| B3 (img-only FT)       | test_dd  | 0.9880 | 0.9841 | 0.9920 | 0.9880 |
| Zero-shot text         | test_dd  | 0.6060 | 1.0000 | 0.2120 | 0.3498 |
| Zero-shot img+text     | test_dd  | 0.6020 | 1.0000 | 0.2040 | 0.3389 |

## rust_java  (未见语言对)

| Model                  | Split    |    Acc |   Prec |    Rec |     F1 |
|------------------------|----------|--------|--------|--------|--------|
| B1 (text FT)           | test_sd  | 0.9920 | 0.9940 | 0.9900 | 0.9920 |
| B1ctrl (blk+text FT)   | test_sd  | 0.9950 | 0.9960 | 0.9940 | 0.9950 |
| B2 (img+text FT)       | test_sd  | 0.9940 | 0.9940 | 0.9940 | 0.9940 |
| B3 (img-only FT)       | test_sd  | 0.9770 | 0.9897 | 0.9640 | 0.9767 |
| Zero-shot text         | test_sd  | 0.6060 | 1.0000 | 0.2120 | 0.3498 |
| Zero-shot img+text     | test_sd  | 0.6130 | 1.0000 | 0.2260 | 0.3687 |
| B1 (text FT)           | test_dd  | 0.9910 | 0.9881 | 0.9940 | 0.9910 |
| B1ctrl (blk+text FT)   | test_dd  | 0.9920 | 0.9862 | 0.9980 | 0.9920 |
| B2 (img+text FT)       | test_dd  | 0.9890 | 0.9822 | 0.9960 | 0.9891 |
| B3 (img-only FT)       | test_dd  | 0.9830 | 0.9820 | 0.9840 | 0.9830 |
| Zero-shot text         | test_dd  | 0.6440 | 0.9932 | 0.2900 | 0.4489 |
| Zero-shot img+text     | test_dd  | 0.6500 | 0.9934 | 0.3020 | 0.4632 |

## rust_python  (未见语言对)

| Model                  | Split    |    Acc |   Prec |    Rec |     F1 |
|------------------------|----------|--------|--------|--------|--------|
| B1 (text FT)           | test_sd  | 0.9870 | 0.9860 | 0.9880 | 0.9870 |
| B1ctrl (blk+text FT)   | test_sd  | 0.9890 | 0.9900 | 0.9880 | 0.9890 |
| B2 (img+text FT)       | test_sd  | 0.9890 | 0.9880 | 0.9900 | 0.9890 |
| B3 (img-only FT)       | test_sd  | 0.9740 | 0.9817 | 0.9660 | 0.9738 |
| Zero-shot text         | test_sd  | 0.6090 | 1.0000 | 0.2180 | 0.3580 |
| Zero-shot img+text     | test_sd  | 0.6240 | 1.0000 | 0.2480 | 0.3974 |
| B1 (text FT)           | test_dd  | 0.9890 | 0.9880 | 0.9900 | 0.9890 |
| B1ctrl (blk+text FT)   | test_dd  | 0.9910 | 0.9900 | 0.9920 | 0.9910 |
| B2 (img+text FT)       | test_dd  | 0.9890 | 0.9861 | 0.9920 | 0.9890 |
| B3 (img-only FT)       | test_dd  | 0.9710 | 0.9738 | 0.9680 | 0.9709 |
| Zero-shot text         | test_dd  | 0.6200 | 1.0000 | 0.2400 | 0.3871 |
| Zero-shot img+text     | test_dd  | 0.6370 | 0.9928 | 0.2760 | 0.4319 |

## rust_ruby  (未见语言对)

| Model                  | Split    |    Acc |   Prec |    Rec |     F1 |
|------------------------|----------|--------|--------|--------|--------|
| B1 (text FT)           | test_sd  | 0.9920 | 0.9920 | 0.9920 | 0.9920 |
| B1ctrl (blk+text FT)   | test_sd  | 0.9940 | 0.9940 | 0.9940 | 0.9940 |
| B2 (img+text FT)       | test_sd  | 0.9960 | 0.9960 | 0.9960 | 0.9960 |
| B3 (img-only FT)       | test_sd  | 0.9760 | 0.9857 | 0.9660 | 0.9758 |
| Zero-shot text         | test_sd  | 0.6130 | 1.0000 | 0.2260 | 0.3687 |
| Zero-shot img+text     | test_sd  | 0.6220 | 1.0000 | 0.2440 | 0.3923 |
| B1 (text FT)           | test_dd  | 0.9910 | 0.9842 | 0.9980 | 0.9911 |
| B1ctrl (blk+text FT)   | test_dd  | 0.9880 | 0.9803 | 0.9960 | 0.9881 |
| B2 (img+text FT)       | test_dd  | 0.9880 | 0.9822 | 0.9940 | 0.9881 |
| B3 (img-only FT)       | test_dd  | 0.9750 | 0.9817 | 0.9680 | 0.9748 |
| Zero-shot text         | test_dd  | 0.6040 | 0.9906 | 0.2100 | 0.3465 |
| Zero-shot img+text     | test_dd  | 0.6080 | 0.9909 | 0.2180 | 0.3574 |
