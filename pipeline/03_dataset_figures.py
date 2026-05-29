#!/usr/bin/env python3
"""
03_dataset_figures.py
Generate dataset statistics figures after pair construction and image rendering.

Outputs (in /data1/clone-test/figures/):
  06_dataset_overview.png        - pair counts per language pair × split
  07_length_stratification.png   - line-count bucket distribution in sampled pairs
  08_overlap_problems.png        - # overlap problems per pair and pool
  09_font_size_heatmap.png       - actual font sizes used in rendered images (sampled)
  10_example_renders.png         - sample rendered images (2 per language)
"""

import json, random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

DATASET_DIR = Path("/data1/clone-test/dataset")
SPLITS_DIR  = DATASET_DIR / "splits"
IMAGES_DIR  = DATASET_DIR / "images"
FIGURES_DIR = Path("/data1/clone-test/figures")
FIGURES_DIR.mkdir(exist_ok=True)

LANG_PAIRS = [
    ("Python", "Java"),
    ("Python", "C++"),
    ("Python", "Ruby"),
    ("Java",   "Ruby"),
    ("Rust",   "Java"),
    ("Rust",   "Python"),
    ("Rust",   "Ruby"),
    ("Kotlin", "Java"),
    ("Scala",  "Java"),
]

TIER_COLORS = {
    ("Python", "Java"):   "#4CAF50",
    ("Python", "C++"):    "#66BB6A",
    ("Python", "Ruby"):   "#2196F3",
    ("Java",   "Ruby"):   "#42A5F5",
    ("Rust",   "Java"):   "#FF9800",
    ("Rust",   "Python"): "#FFA726",
    ("Rust",   "Ruby"):   "#FFB74D",
    ("Kotlin", "Java"):   "#E91E63",
    ("Scala",  "Java"):   "#EC407A",
}

SPLIT_COLORS = {
    "train_a": "#1B5E20",
    "train_b": "#388E3C",
    "val":     "#1565C0",
    "test_sd": "#E65100",
    "test_dd": "#BF360C",
    "zeroshot_w1": "#4A148C",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

SEED = 42
rng = random.Random(SEED)


def pair_key(l1, l2):
    return f"{l1.lower().replace('+','p')}_{l2.lower()}"


def load_summary():
    p = DATASET_DIR / "dataset_summary.json"
    if not p.exists():
        return {}
    return json.load(open(p))


def load_splits_for_pair(l1, l2, split_names):
    """Load pairs from specified split JSONL files for a language pair."""
    pk = pair_key(l1, l2)
    pairs = {}
    for sname in split_names:
        path = SPLITS_DIR / pk / f"{sname}.jsonl"
        if not path.exists():
            continue
        ps = []
        with open(path) as f:
            for line in f:
                try:
                    ps.append(json.loads(line))
                except Exception:
                    pass
        pairs[sname] = ps
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Dataset overview: pair counts per pair × split
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dataset_overview():
    summary = load_summary()
    if not summary:
        print("  SKIP fig 6: no dataset_summary.json")
        return

    all_splits = ["train_a", "train_b", "val", "test_sd", "test_dd", "zeroshot_w1"]
    pairs_labels = [f"{l1}↔{l2}" for l1, l2 in LANG_PAIRS]

    data = np.zeros((len(all_splits), len(LANG_PAIRS)), dtype=int)
    for j, (l1, l2) in enumerate(LANG_PAIRS):
        pk = pair_key(l1, l2)
        if pk not in summary:
            continue
        s = summary[pk]
        for i, sp in enumerate(all_splits):
            data[i, j] = s.get(sp, 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(LANG_PAIRS))
    width = 0.13
    offsets = np.linspace(-(len(all_splits)-1)/2, (len(all_splits)-1)/2, len(all_splits)) * width

    for i, sp in enumerate(all_splits):
        bars = ax.bar(x + offsets[i], data[i], width,
                      label=sp, color=SPLIT_COLORS[sp], alpha=0.85)
        for bar, val in zip(bars, data[i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f"{val//1000}K" if val >= 1000 else str(val),
                        ha="center", va="bottom", fontsize=7, rotation=70)

    ax.set_xticks(x)
    ax.set_xticklabels(pairs_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Number of pairs")
    ax.set_title("Dataset Pair Counts per Language Pair × Split\n(pos+neg, 1:1 balanced)")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # Tier separators
    ax.axvline(1.5, color="#aaa", lw=1, ls="--", alpha=0.5)  # high→mid
    ax.axvline(3.5, color="#aaa", lw=1, ls="--", alpha=0.5)  # mid→low
    ax.axvline(6.5, color="#aaa", lw=1, ls="--", alpha=0.5)  # low→OOD
    ax.text(0.75, ax.get_ylim()[1]*0.95, "High-res", ha="center", fontsize=9, color="#4CAF50")
    ax.text(2.5,  ax.get_ylim()[1]*0.95, "Mid-res",  ha="center", fontsize=9, color="#2196F3")
    ax.text(5.0,  ax.get_ylim()[1]*0.95, "Low-res",  ha="center", fontsize=9, color="#FF9800")
    ax.text(7.5,  ax.get_ylim()[1]*0.95, "OOD",      ha="center", fontsize=9, color="#E91E63")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_dataset_overview.png")
    plt.close(fig)
    print("  Saved 06_dataset_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Length stratification in sampled pairs
# ═══════════════════════════════════════════════════════════════════════════════
LENGTH_BUCKETS = [(1, 15), (16, 30), (31, 50), (51, 200)]
BUCKET_LABELS  = ["1–15 L", "16–30 L", "31–50 L", "51+ L"]

def bucket_of(n):
    for i, (lo, hi) in enumerate(LENGTH_BUCKETS):
        if lo <= n <= hi:
            return i
    return len(LENGTH_BUCKETS) - 1

def fig_length_stratification():
    # Load test_sd pairs for all language pairs (representative sample)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes_flat = axes.flatten()

    for idx, (l1, l2) in enumerate(LANG_PAIRS):
        ax = axes_flat[idx]
        pk = pair_key(l1, l2)
        jsonl = SPLITS_DIR / pk / "test_sd.jsonl"
        if not jsonl.exists():
            ax.set_visible(False)
            continue

        max_lens = []
        with open(jsonl) as f:
            for line in f:
                try:
                    p = json.loads(line)
                    ml = max(p["code1"].get("n_lines", 0), p["code2"].get("n_lines", 0))
                    max_lens.append(ml)
                except Exception:
                    pass

        if not max_lens:
            ax.set_visible(False)
            continue

        buckets = [0] * len(LENGTH_BUCKETS)
        for ml in max_lens:
            buckets[bucket_of(ml)] += 1
        total = sum(buckets)

        color = TIER_COLORS[(l1, l2)]
        bars = ax.bar(BUCKET_LABELS, [b/total*100 for b in buckets],
                      color=color, alpha=0.8, edgecolor="white", lw=0.5)
        for bar, b in zip(bars, buckets):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(b), ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{l1} ↔ {l2}\n(test_sd, n={total})", fontsize=10)
        ax.set_ylabel("% of pairs")
        ax.set_ylim(0, 55)
        ax.axhline(25, color="#ccc", lw=1, ls="--", alpha=0.7, label="uniform 25%")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Length-Stratified Sampling — Pair Distribution by max(len1, len2)\n"
                 "Dashed line = perfect uniform distribution (25% per bucket)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_length_stratification.png")
    plt.close(fig)
    print("  Saved 07_length_stratification.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Overlap problem counts per pair and pool
# ═══════════════════════════════════════════════════════════════════════════════
def fig_overlap_problems():
    problem_splits = {}
    p = DATASET_DIR / "problem_splits.json"
    if p.exists():
        problem_splits = json.load(open(p))
    else:
        print("  SKIP fig 8: no problem_splits.json")
        return

    # Count overlap per pair per pool by checking actual JSONL files
    summary = load_summary()
    if not summary:
        print("  SKIP fig 8: no dataset_summary.json")
        return

    pair_labels = [f"{l1}↔{l2}" for l1, l2 in LANG_PAIRS]
    # Estimate overlap from summary: test_sd / 2 ≈ n_overlap_train (very rough)
    # Instead, count unique problem_ids from test files

    def count_unique_problems(l1, l2, split):
        pk = pair_key(l1, l2)
        jsonl = SPLITS_DIR / pk / f"{split}.jsonl"
        if not jsonl.exists():
            return 0
        pids = set()
        with open(jsonl) as f:
            for line in f:
                try:
                    pair = json.loads(line)
                    if pair["label"] == 1:  # only positives have meaningful problem info
                        pids.add(pair["code1"]["problem_id"])
                except Exception:
                    pass
        return len(pids)

    train_overlaps = [count_unique_problems(l1, l2, "test_sd") for l1, l2 in LANG_PAIRS]
    val_overlaps   = [count_unique_problems(l1, l2, "val")     for l1, l2 in LANG_PAIRS]
    test_overlaps  = [count_unique_problems(l1, l2, "test_dd") for l1, l2 in LANG_PAIRS]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(LANG_PAIRS))
    w = 0.28

    b1 = ax.bar(x - w, train_overlaps, w, label="train_pool (SD)", color="#388E3C", alpha=0.85)
    b2 = ax.bar(x,     val_overlaps,   w, label="val_pool",         color="#1565C0", alpha=0.85)
    b3 = ax.bar(x + w, test_overlaps,  w, label="test_pool (DD)",   color="#BF360C", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, str(h),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=30, ha="right")
    ax.set_ylabel("Unique problems with overlap (in positive pairs)")
    ax.set_title("Overlap Problem Counts per Language Pair and Pool\n"
                 "(estimated from positive pairs in each split file)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_overlap_problems.png")
    plt.close(fig)
    print("  Saved 08_overlap_problems.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Font size distribution across rendered images (sampled)
# ═══════════════════════════════════════════════════════════════════════════════
FONT_STEPS = [24, 22, 20, 18, 16, 14, 12, 11, 10, 9]
PAD = 48
CHAR_W_F = 0.8
LINE_H_F = 1.6

def pick_font(n_lines, max_chars):
    for portrait in [False, True]:
        H = 1120 if portrait else 896
        W = 896 if portrait else 1120
        for font in FONT_STEPS:
            if (n_lines * LINE_H_F * font + PAD <= H and
                    max_chars * CHAR_W_F * font + PAD <= W):
                return font
    return 9

def fig_font_size_heatmap():
    # Sample pairs from all splits and compute font sizes
    lang_font_data = defaultdict(list)
    for l1, l2 in LANG_PAIRS:
        pk = pair_key(l1, l2)
        for split in ["test_sd", "test_dd", "train_a", "train_b"]:
            jsonl = SPLITS_DIR / pk / f"{split}.jsonl"
            if not jsonl.exists():
                continue
            with open(jsonl) as f:
                lines = f.readlines()
            sample = rng.sample(lines, min(50, len(lines)))
            for line in sample:
                try:
                    pair = json.loads(line)
                    for key in ("code1", "code2"):
                        c = pair[key]
                        nl = c.get("n_lines", 0)
                        mc = c.get("max_chars", 0)
                        if nl > 0 and mc > 0:
                            fs = pick_font(nl, mc)
                            lang_font_data[c["lang"]].append(fs)
                except Exception:
                    pass

    if not lang_font_data:
        print("  SKIP fig 9: no pair data")
        return

    langs = ["Python", "Java", "C++", "Ruby", "Rust", "Kotlin", "Scala"]
    font_vals = sorted(set(FONT_STEPS))

    matrix = np.zeros((len(langs), len(font_vals)))
    for i, lang in enumerate(langs):
        data = lang_font_data.get(lang, [])
        if data:
            for v in data:
                if v in font_vals:
                    j = font_vals.index(v)
                    matrix[i, j] += 1
            matrix[i] = matrix[i] / matrix[i].sum() * 100  # normalize to %

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% of code snippets")

    ax.set_xticks(range(len(font_vals)))
    ax.set_xticklabels([f"{v}pt" for v in font_vals])
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    ax.set_xlabel("Font size used for rendering")
    ax.set_title("Font Size Distribution per Language\n(based on sampled pairs)")

    for i in range(len(langs)):
        for j in range(len(font_vals)):
            val = matrix[i, j]
            if val > 3:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, color="black" if val < 60 else "white")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_font_size_heatmap.png")
    plt.close(fig)
    print("  Saved 09_font_size_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Sample rendered images (1 positive pair per tier)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_example_renders():
    sample_pairs = [
        ("Python", "Java",   "train_a"),   # high-resource
        ("Rust",   "Java",   "test_sd"),   # low-resource
        ("Kotlin", "Java",   "test_dd"),   # OOD
    ]

    fig, axes = plt.subplots(len(sample_pairs), 2, figsize=(16, 12))
    fig.suptitle("Sample Rendered Code Images (positive clone pairs)", fontsize=14)

    for row_idx, (l1, l2, split) in enumerate(sample_pairs):
        pk = pair_key(l1, l2)
        jsonl = SPLITS_DIR / pk / f"{split}.jsonl"
        if not jsonl.exists():
            for col in range(2):
                axes[row_idx, col].axis("off")
                axes[row_idx, col].set_title(f"{l1}↔{l2} [{split}] NOT FOUND")
            continue

        # Find first positive pair with both images rendered
        with open(jsonl) as f:
            lines = f.readlines()
        rng.shuffle(lines)

        found = False
        for line in lines[:50]:
            try:
                pair = json.loads(line)
            except Exception:
                continue
            if pair["label"] != 1:
                continue
            img1_path = IMAGES_DIR / pair["code1"]["image_rel_path"].replace("images/", "")
            img2_path = IMAGES_DIR / pair["code2"]["image_rel_path"].replace("images/", "")
            if img1_path.exists() and img2_path.exists():
                img1 = Image.open(img1_path)
                img2 = Image.open(img2_path)
                found = True
                break

        if not found:
            for col in range(2):
                axes[row_idx, col].axis("off")
                axes[row_idx, col].set_title(f"{l1}↔{l2} images not yet rendered")
            continue

        axes[row_idx, 0].imshow(img1)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(
            f"{l1} (problem {pair['code1']['problem_id']}, {pair['code1']['n_lines']} lines)",
            fontsize=10)

        axes[row_idx, 1].imshow(img2)
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title(
            f"{l2} (problem {pair['code2']['problem_id']}, {pair['code2']['n_lines']} lines)",
            fontsize=10)

        # Tier label on the side
        fig.text(0.01, axes[row_idx, 0].get_position().y0 + 0.1,
                 f"{'High' if 'Java' in (l1,l2) and 'Python' in (l1,l2) else 'Low' if 'Rust' in (l1,l2) else 'OOD'}",
                 fontsize=10, va="center", color="gray", rotation=90)

    fig.tight_layout(rect=[0.03, 0, 1, 1])
    fig.savefig(FIGURES_DIR / "10_example_renders.png")
    plt.close(fig)
    print("  Saved 10_example_renders.png")


def main():
    print("Generating dataset figures...")
    print()
    fig_dataset_overview()
    fig_length_stratification()
    fig_overlap_problems()
    fig_font_size_heatmap()
    fig_example_renders()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
