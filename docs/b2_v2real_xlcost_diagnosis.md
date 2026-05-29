# B2-v2real 在 XLCoST 上的退化：现象分析与修复方案

**日期**：2026-05-25
**任务**：诊断 3.5-B2 fine-tuned on real v2 data（含 HARD_NEG）在 XLCoST 跨语言克隆检测上表现差于 zero-shot B2 与 B1-v2real 的原因。
**结论**：B2 微调诱导了**视觉模态过度依赖**，导致 OOD 图像分布下模型决策从"激进"翻转为"极端保守"，recall 从 0.901（同分布）崩塌到 0.366（OOD）。

---

## 1. 现象：F1 对比矩阵

### 1.1 XLCoST F1（6 语言对均值，OOD 测试）

| 模型 | XLCoST Avg F1 | 备注 |
|---|---|---|
| 3.5 ZS-B2（无 FT） | 0.774 | baseline |
| 3.5 B2-old（v1 训练，label-leakage） | 0.001 | 历史 bug，已修复 |
| **3.5 B2-v2real（v1 修复后）** | **0.535** | ⚠️ **比 ZS 还低 0.24** |
| 3.5 B1-v2real（text-only FT） | 0.938 | 同训练数据，无图像 |

### 1.2 同分布对照

| 模型 | v2 val F1 | CodeNet test_sd F1 | CodeNet test_dd F1 |
|---|---|---|---|
| 3.5 B2-v2real | 0.805 | 0.868 | 0.867 |

✅ **模型在同分布上完全健康**。问题专属于 OOD。

### 1.3 Precision / Recall 完整分解

| 模型 / 测试集 | F1 | Precision | Recall | 决策行为 |
|---|---|---|---|---|
| B2-v2real / v2 val（in-dist） | 0.805 | 0.727 | **0.901** | 激进说 Yes |
| B2-v2real / XLCoST avg（OOD） | 0.535 | **1.000** | **0.366** | 极端保守 |
| ZS-B2 / XLCoST avg | 0.774 | 1.000 | 0.632 | 中等保守 |
| B1-v2real / XLCoST avg | 0.938 | 1.000 | 0.883 | 几乎全对 |

**关键反转**：B2-v2real 的 recall 在同分布与 OOD 之间相差 0.535（0.901 → 0.366）。所有模型在 XLCoST 上 Precision=1.0（不存在 FP 问题），失败全部集中在 recall 端。

---

## 2. 决定性证据：per-example 三方对比

在 XLCoST java_python 同一批 200 个测试样本上保存逐条预测：

```
gold  ZS-B2  B2-v2real  B1-v2real    count
  1     1       1          1          38   ← 三模型都对
  1     1       0          1          28   ← FT 把对的改错了
  1     0       0          1          25   ← B1 救回来
  1     0       0          0           9   ← 三模型都错
  1     0       1          1           4
  1     1       0          0           1
  0     0       0          0          95   ← 负样本全对
```

**POS recall**：ZS-B2 = 0.638；**B2-v2real = 0.400**；**B1-v2real = 0.905**

**最关键的数字**：在 B2-v2real 答错的 63 个 POS pair 中，B1-v2real 用**完全相同的训练数据 + 完全相同的文本输入**正确识别 **84%（53/63）**。

唯一变量是图像输入。**图像在主动误导 B2-v2real**。

---

## 3. 失败机制分析

### 3.1 训练阶段：模态权重学习

`runs/v2real_b2_train_a_seed42/adapter_config.json` 显示：
- 160 个 LoRA 模块**全部在 `language_model.layers.*` 内**
- 视觉编码器（`Qwen3_5VisionModel`）完全冻结（`no_grad`）
- LoRA target: `q_proj k_proj v_proj o_proj in_proj_qkv out_proj`（其中 `in_proj_qkv/out_proj` 是 Qwen3.5 的 linear_attn 层，**不是视觉投影**）

因此 B2-v2real 与 ZS-B2 处理图像的视觉通路**完全相同**；差异仅在 LM 如何**利用图像 embedding**。

训练数据 20k pairs（50% POS + 25% HARD_NEG + 25% EASY_NEG）的 in-distribution 上，图像与标签紧密耦合：
- 同 problem 的 AC×AC 视觉风格高度相似（POS）
- 同 problem 的 AC×WA 视觉风格相似但代码逻辑微差（HARD_NEG）
- 跨 problem 视觉差异大（EASY_NEG）

LM 通过 LoRA 学到："图像 embedding 的相似度模式是判定 clone 的强证据"。

### 3.2 OOD 阶段：分布偏移破坏决策

XLCoST 图像虽然渲染样式相同（Pygments Monokai、1440×896、相同字体），但内容布局分布偏移：

1. **结构层**：Java 全部用 `class GFG {}` 包装器，CodeNet 多为 `public class Main {}` 或多类结构；命名空间与缩进特征都不同。
2. **长度层**：XLCoST 是 GeeksforGeeks 教学代码，平均更短；图像下半多空白。
3. **内容层**：去 token 化伪影使 Python 出现 `if __name__ == " _ _ main _ _ ":`、末尾 `;`、`' a '` 字符字面量等 OOD token 模式。

冻结的视觉编码器仍能产出有效 embedding，但其分布与训练时偏移。LM 经 LoRA 适配后，对这种分布偏移**特别敏感**——它学到的"看起来像 clone"特征签名失效，模型转向默认拒绝。

### 3.3 为什么 B1 不受影响

| 模式 | 输入信号 | OOD 泛化路径 |
|---|---|---|
| B1 | 仅文本 | 文本中的代码逻辑**直接可读**，即使有去 token 化伪影，控制流仍清晰 → 文本推理通路在 OOD 仍可工作 |
| B2 | 文本 + 图像 | FT 期间图像权重被抬高（in-dist 上图像信号最强），文本退化为辅助证据 → OOD 上图像失效时，模型不会回退到文本通路 |

这是经典的**模态崩塌（modal collapse）**：当一个模态在训练时主导决策，模型在该模态失效时不会自动切换到次要模态。

### 3.4 为什么失败全在 recall 端

- **Negative pair（gold=0）**：本身视觉差异大，"说 No"不需要强证据 → 不论分布是否偏移，模型都正确说 No → TNR=1.0
- **Positive pair（gold=1）**：需要图像 embedding 命中"clone 特征签名"才说 Yes。OOD 图像无法命中签名 → 默认 No → recall 崩塌

这也解释了 B2-old（label-leakage 训练）的 F1=0：v1 训练数据中正样本图像 100% 同 problem，模型学到极强的"图像视觉相似 → clone"捷径；XLCoST Java vs Python 视觉显著不对称（GFG wrapper vs 裸脚本）→ 100% 拒绝。

B2-v2real 用 HARD_NEG 弱化了这条捷径，所以从 0.001 恢复到 0.535，但仍未达到 ZS 水平。

---

## 4. 修复方案

按优先级与可行性排序：

### P0：图像增广训练（推荐先尝试）

随机化训练时图像的视觉属性，迫使 LM 不能依赖单一渲染分布：

```python
# 在 dataloader collator 内随机选择：
augmentations = [
    "font_size in {14, 16, 18, 20}",   # 字号
    "color_scheme in [monokai, dracula, github_dark, vim]",
    "font_name in [DejaVu Sans Mono, Fira Code, JetBrains Mono]",
    "image_pad in {5, 10, 15, 20}",
    "line_pad in {2, 3, 4, 5}",
]
```

**预期效果**：模型学到"代码语义不应受渲染风格影响"，OOD 鲁棒性显著提升。

**成本**：需要重训 B2（约 17h），但训练阶段渲染开销可接受（每 epoch 一次或预渲染多版本）。

### P1：image-dropout 训练

每个 batch 以概率 p=0.3 把图像替换为纯背景（黑色 1440×896 PNG）：

```python
if random.random() < 0.3:
    images = [Image.new("RGB", (1440,896), (39,40,34))] * len(images)
```

模型被迫学习"图像缺失时也能从文本判定"，建立 B1+B2 双通路的决策能力。本质上是让 B2 行为退化到 B1 行为作为 fallback。

**预期效果**：OOD 上即使图像失效，文本通路也能给出正确答案。

### P2：降低图像贡献的结构调整

- **缩短图像 token 数量**：在 processor 配置里降低图像 tile 数（如 256→128 patches），减少图像在序列中的相对权重。
- **不要在 vision-LM cross-attention 上加 LoRA**：当前 LoRA 在所有 `language_model.layers` 的 attention 上都加了 LoRA，跨模态融合层也被微调。可改为只在 self-attention 层加 LoRA，cross-attention 保持原始权重。

### P3：训练数据多样化

加入第二个代码源（如 LeetCode、SPOJ 或 GeeksforGeeks）作为训练辅助。即使每个源 5k pairs，多分布训练也能显著降低对 CodeNet 单一分布的依赖。

**注意**：加入 GeeksforGeeks 本身相当于 XLCoST 数据泄露，需要排除 XLCoST 测试集涉及的题目。

### P4：超参微调

- **降低 LoRA rank**：当前 r=16，改为 r=8 可降低过拟合到 CodeNet 视觉分布的风险。
- **缩短 epochs**：当前 3 epochs；改为 2 epochs 或 early stopping by OOD val（例如保留一小份 GPTCloneBench 作 OOD val）。
- **更低学习率**：从 1e-4 降到 5e-5，减少 LoRA 的 aggressiveness。

### P5（探索性）：训练目标改进

在 SFT loss 之外加入**模态一致性约束**：

```
L = L_SFT + λ * KL(p_b2 || p_b1_with_same_input)
```

让 B2 模式（有图）的输出分布与 B1 模式（无图）的输出分布接近，强制图像作为"次要证据"而非"主要决策器"。

---

## 5. 实验设计建议

为快速验证哪个方案有效，建议按以下顺序做小规模 ablation（每个用 5k pairs 训练 1 epoch，~3h）：

| 实验 | 训练配置 | 验证指标 |
|---|---|---|
| A0（baseline 复现） | 当前 v2real 配置 | XLCoST F1 应 ≈ 0.535 |
| A1（image-dropout） | + p=0.3 image dropout | XLCoST F1 是否 > 0.7 |
| A2（颜色+字体增广） | + Monokai/Dracula/GitHub 三选一，字体 3 选一 | XLCoST F1 是否 > 0.7 |
| A3（A1 + A2 组合） | image-dropout + 增广 | OOD F1 上限 |
| A4（更低 rank） | r=8, 其他不变 | 是否轻微改善 |

每个实验只需 evaluate XLCoST avg F1 + CodeNet test_sd F1 即可比较。

---

## 6. 临时缓解措施

如果短期内无法重训，可考虑：

1. **使用 B1-v2real 做主分类，B2-v2real 仅作为辅助置信度信号**——B1 在 OOD 已经达到 0.938。
2. **Ensemble**：B1-v2real 与 B2-v2real 多数投票，或在 B2 说 No 时 fallback 到 B1。
3. **测试时图像归一化**：对 OOD 测试集的图像做风格迁移（neural style transfer）到 CodeNet 训练风格——可行性低，作为下下策。

---

## 附录：原始数据

- B2-v2real checkpoint: `runs/v2real_b2_train_a_seed42/`
- 训练脚本: `finetune/run_train_b2_v2real.sh`
- 训练数据: `dataset/finetune_data_v2/b2/train_a/python_java.jsonl`（20k pairs，hard/easy neg 各 25%）
- 评估结果: `results/v2real/qw35_b2v2real_*.json`
- 诊断 per-example 预测: `results/v2real/diag/{zs_b2,v2real_b2,v2real_b1}_preds.jsonl`
- val 评估: `results/v2real/qw35_b2v2real_val.json`
