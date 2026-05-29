# Prompt Ablation: Default vs. Ian-Kappa Style

**Models:**
- `Coder` = Qwen2.5-Coder-3B-Instruct
- `3.5`   = Qwen3.5-4B

**Conditions:**
- `ZS` = Zero-shot (text only, B1 format)
- `ZS-B2` = Zero-shot (image + text, B2 format)
- `B1` = Fine-tuned on v2 data, text-only
- `B2` = Fine-tuned on v2 data, image + text

**Prompt styles:**
- **Orig**: simple Yes/No instruction, no clone definition
- **IK**: ian-Kappa style with detailed semantic clone definition tailored to our dataset construction logic
- **Δ**: IK − Orig (positive = improvement)

## test_SD (Submission-Disjoint)

| Language Pair | Coder-ZS Orig | Coder-ZS IK | Coder-ZS Δ | Coder-B1 Orig | Coder-B1 IK | Coder-B1 Δ | 3.5-ZS Orig | 3.5-ZS IK | 3.5-ZS Δ | 3.5-ZS-B2 Orig | 3.5-ZS-B2 IK | 3.5-ZS-B2 Δ | 3.5-B1 Orig | 3.5-B1 IK | 3.5-B1 Δ | 3.5-B2 Orig | 3.5-B2 IK | 3.5-B2 Δ |
| :------------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python↔Java | 0.556 | 0.627 | **+0.071** | 0.786 | 0.763 | **-0.024** | 0.372 | 0.579 | **+0.208** | 0.371 | 0.620 | **+0.248** | 0.865 | 0.866 | **+0.001** | 0.988 | 0.982 | **-0.006** |
| Python↔C++ | 0.593 | 0.602 | **+0.009** | 0.909 | 0.899 | **-0.010** | 0.390 | 0.392 | **+0.003** | 0.395 | 0.395 | **+0.000** | 0.956 | 0.957 | **+0.001** | 0.981 | 0.981 | **+0.000** |
| Python↔Ruby | 0.770 | 0.761 | **-0.010** | 0.905 | 0.974 | **+0.069** | 0.288 | 0.571 | **+0.284** | 0.314 | 0.702 | **+0.389** | 0.953 | 0.951 | **-0.002** | 0.975 | 0.986 | **+0.010** |
| Java↔Ruby | 0.677 | 0.766 | **+0.089** | 0.886 | 0.934 | **+0.048** | 0.382 | 0.580 | **+0.198** | 0.366 | 0.692 | **+0.326** | 0.910 | 0.901 | **-0.009** | 0.954 | 0.978 | **+0.023** |
| **AVG** | **0.649** | **0.689** | **+0.040** | **0.871** | **0.892** | **+0.021** | **0.358** | **0.531** | **+0.173** | **0.361** | **0.602** | **+0.241** | **0.921** | **0.919** | **-0.002** | **0.975** | **0.982** | **+0.007** |

## test_DD (Problem-Disjoint)

| Language Pair | Coder-ZS Orig | Coder-ZS IK | Coder-ZS Δ | Coder-B1 Orig | Coder-B1 IK | Coder-B1 Δ | 3.5-ZS Orig | 3.5-ZS IK | 3.5-ZS Δ | 3.5-ZS-B2 Orig | 3.5-ZS-B2 IK | 3.5-ZS-B2 Δ | 3.5-B1 Orig | 3.5-B1 IK | 3.5-B1 Δ | 3.5-B2 Orig | 3.5-B2 IK | 3.5-B2 Δ |
| :------------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python↔Java | 0.621 | 0.752 | **+0.131** | 0.759 | 0.781 | **+0.023** | 0.349 | 0.599 | **+0.250** | 0.384 | 0.676 | **+0.292** | 0.842 | 0.835 | **-0.007** | 0.994 | 0.985 | **-0.009** |
| Python↔C++ | 0.594 | 0.500 | **-0.094** | 0.860 | 0.822 | **-0.039** | 0.369 | 0.363 | **-0.005** | 0.377 | 0.377 | **+0.000** | 0.959 | 0.959 | **+0.000** | 0.982 | 0.982 | **+0.000** |
| Python↔Ruby | 0.776 | 0.797 | **+0.021** | 0.864 | 0.950 | **+0.086** | 0.255 | 0.527 | **+0.272** | 0.288 | 0.719 | **+0.432** | 0.944 | 0.951 | **+0.007** | 0.972 | 0.983 | **+0.010** |
| Java↔Ruby | 0.698 | 0.867 | **+0.168** | 0.892 | 0.960 | **+0.068** | 0.285 | 0.542 | **+0.258** | 0.331 | 0.686 | **+0.355** | 0.929 | 0.926 | **-0.003** | 0.974 | 0.985 | **+0.010** |
| **AVG** | **0.672** | **0.729** | **+0.056** | **0.844** | **0.878** | **+0.035** | **0.314** | **0.508** | **+0.194** | **0.345** | **0.614** | **+0.270** | **0.918** | **0.918** | **-0.001** | **0.981** | **0.984** | **+0.003** |

## Key Findings

1. **Zero-shot gains most from IK prompt**: 3.5-ZS improves by ~+0.17–0.19 F1 on average; 3.5-ZS-B2 by ~+0.24–0.27. The detailed clone definition gives the model a clear decision boundary it otherwise lacks.
2. **Fine-tuned models are prompt-invariant**: 3.5-B1 and 3.5-B2 show Δ ≈ 0.00, indicating that fine-tuning internalises the judgment criteria regardless of prompt wording.
3. **Coder ZS benefits modestly** (+0.04–0.06): Coder's stronger code pretraining provides a partial substitute for an explicit definition, so the marginal gain is smaller.
4. **Coder B1-FT gains slightly** (SD +0.02, DD +0.03): A small consistent improvement suggests the IK definition helps even fine-tuned text-only models at the margin.