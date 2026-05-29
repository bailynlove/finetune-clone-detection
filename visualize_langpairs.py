#!/usr/bin/env python3
"""
Generate visualization images for the selected CodeNet language pairs
based on finetune_plan.md and the actual distribution stats from codenet_stats.json.

Outputs:
  figures/01_line_count_dist.png      - box-style distribution of line counts
  figures/02_maxlinelen_dist.png      - box-style distribution of max line lengths
  figures/03_lang_pair_tiers.png      - language pair tier structure
  figures/04_font_size_guide.png      - data-driven font size recommendation
"""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
STATS_PATH = Path("/data1/clone-test/codenet_stats.json")
OUT_DIR    = Path("/data1/clone-test/figures")
OUT_DIR.mkdir(exist_ok=True)

# Target languages (from finetune_plan.md §2.2)
TARGET_LANGS = ["Python", "Java", "C++", "Ruby", "Rust", "Kotlin", "Scala"]

# Language pair tiers
TIERS = {
    "High-resource\n(Train: main)":       [("Python", "Java"), ("Python", "C++")],
    "Mid-resource\n(Train: zero-shot)":   [("Python", "Ruby"), ("Java", "Ruby")],
    "Low-resource\n(Train: variant B)":   [("Rust", "Java"), ("Rust", "Python"), ("Rust", "Ruby")],
    "OOD\n(Train: strict zero-shot)":     [("Kotlin", "Java"), ("Scala", "Java")],
}

TIER_COLORS = {
    "High-resource\n(Train: main)":      "#4CAF50",   # green
    "Mid-resource\n(Train: zero-shot)":  "#2196F3",   # blue
    "Low-resource\n(Train: variant B)":  "#FF9800",   # orange
    "OOD\n(Train: strict zero-shot)":    "#E91E63",   # pink/red
}

LANG_COLORS = {
    "Python": "#3776AB",
    "Java":   "#ED8B00",
    "C++":    "#00599C",
    "Ruby":   "#CC342D",
    "Rust":   "#CE422B",
    "Kotlin": "#7F52FF",
    "Scala":  "#DC322F",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})

# ── load stats ────────────────────────────────────────────────────────────────
with open(STATS_PATH) as f:
    stats = json.load(f)

target_stats = {lang: stats[lang] for lang in TARGET_LANGS}


# ── helper: build box data from percentiles ──────────────────────────────────
def box_from_pcts(s):
    """Return (min, p25, median, p75, p90, p95, max) from a stats dict."""
    return (s["min"], s["p25"], s["median"], s["p75"], s["p90"], s["p95"], s["max"])


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Line count distribution
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

langs = TARGET_LANGS
y_pos = np.arange(len(langs))

for i, lang in enumerate(langs):
    lc = target_stats[lang]["line_count"]
    color = LANG_COLORS[lang]
    total = target_stats[lang]["total_files"]

    # IQR box
    ax.barh(i, lc["p75"] - lc["p25"], left=lc["p25"],
            height=0.55, color=color, alpha=0.75, zorder=3)
    # median line
    ax.plot([lc["median"], lc["median"]], [i - 0.28, i + 0.28],
            color="white", lw=2.5, zorder=4)
    # whiskers to p10 and p90
    ax.plot([lc["min"], lc["p25"]], [i, i], color=color, lw=1.5, zorder=2, alpha=0.6)
    ax.plot([lc["p75"], lc["p90"]], [i, i], color=color, lw=1.5, zorder=2, alpha=0.6)
    # p90 cap
    ax.plot([lc["p90"], lc["p90"]], [i - 0.12, i + 0.12], color=color, lw=1.5, zorder=2, alpha=0.6)
    # p95 dot
    ax.scatter([lc["p95"]], [i], marker="|", s=100, color=color, zorder=5, linewidths=2)

    # annotation: median + total_files
    ax.text(lc["p90"] + 2, i, f" med={lc['median']:.0f}  p90={lc['p90']:.0f}",
            va="center", fontsize=9.5, color="#333")
    ax.text(-4, i, f"{total/1e6:.1f}M", va="center", ha="right", fontsize=9, color="#666")

ax.set_yticks(y_pos)
ax.set_yticklabels(langs, fontsize=12)
ax.set_xlabel("Lines of code (sampled 100 files per language)")
ax.set_title("Line Count Distribution — Selected CodeNet Languages\n"
             "(box=IQR, median=white bar, whisker→p90, tick=p95)", pad=10)
ax.axvline(45, color="#aaa", lw=1, ls="--", label="45-line truncation threshold")
ax.axvline(15, color="#ccc", lw=1, ls=":", label="15-line boundary")
ax.set_xlim(-25, 300)
ax.legend(fontsize=9, loc="lower right")
ax.text(-25, len(langs) - 0.3, "Files(M)", fontsize=9, color="#666")
ax.grid(axis="x", lw=0.4, alpha=0.5)
ax.set_facecolor("#FAFAFA")
fig.tight_layout()
fig.savefig(OUT_DIR / "01_line_count_dist.png")
plt.close(fig)
print("Saved 01_line_count_dist.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Max line length distribution
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

for i, lang in enumerate(langs):
    ml = target_stats[lang]["max_line_len"]
    color = LANG_COLORS[lang]

    ax.barh(i, ml["p75"] - ml["p25"], left=ml["p25"],
            height=0.55, color=color, alpha=0.75, zorder=3)
    ax.plot([ml["median"], ml["median"]], [i - 0.28, i + 0.28],
            color="white", lw=2.5, zorder=4)
    ax.plot([ml["min"], ml["p25"]], [i, i], color=color, lw=1.5, alpha=0.6, zorder=2)
    ax.plot([ml["p75"], ml["p90"]], [i, i], color=color, lw=1.5, alpha=0.6, zorder=2)
    ax.plot([ml["p90"], ml["p90"]], [i - 0.12, i + 0.12], color=color, lw=1.5, alpha=0.6, zorder=2)
    ax.scatter([ml["p95"]], [i], marker="|", s=100, color=color, zorder=5, linewidths=2)
    ax.text(ml["p90"] + 1, i, f" med={ml['median']:.0f}  p90={ml['p90']:.0f}",
            va="center", fontsize=9.5, color="#333")

ax.set_yticks(y_pos)
ax.set_yticklabels(langs, fontsize=12)
ax.set_xlabel("Longest line (characters) in each file")
ax.set_title("Max Line Length Distribution — Selected CodeNet Languages\n"
             "(box=IQR, median=white bar, whisker→p90, tick=p95)", pad=10)
ax.axvline(80,  color="#aaa", lw=1, ls="--", label="80-char convention")
ax.axvline(120, color="#ccc", lw=1, ls=":",  label="120-char soft limit")
ax.set_xlim(0, 230)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="x", lw=0.4, alpha=0.5)
ax.set_facecolor("#FAFAFA")
fig.tight_layout()
fig.savefig(OUT_DIR / "02_maxlinelen_dist.png")
plt.close(fig)
print("Saved 02_maxlinelen_dist.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Language pair tier structure
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis("off")

tier_names = list(TIERS.keys())
n_tiers = len(tier_names)
col_w = 1.0 / n_tiers

for ti, tname in enumerate(tier_names):
    color = TIER_COLORS[tname]
    pairs = TIERS[tname]
    x = ti * col_w + col_w / 2

    # tier header box
    ax.text(x, 0.92, tname.replace("\n", "\n"),
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="none"))

    # resource label
    resource_labels = {
        "High-resource\n(Train: main)":      "Training: YES (main)",
        "Mid-resource\n(Train: zero-shot)":  "Training: NO (zero-shot only)",
        "Low-resource\n(Train: variant B)":  "Training: YES (variant B)",
        "OOD\n(Train: strict zero-shot)":    "Training: NO (strict OOD)",
    }
    ax.text(x, 0.76, resource_labels[tname],
            ha="center", va="center", fontsize=9, color=color,
            style="italic")

    # pairs
    for pi, (l1, l2) in enumerate(pairs):
        y = 0.58 - pi * 0.16
        pair_label = f"{l1} ↔ {l2}"
        # get resource size
        files1 = target_stats[l1]["total_files"]
        files2 = target_stats[l2]["total_files"]
        smaller = min(files1, files2)
        resource_str = f"({smaller/1000:.0f}K files in smaller lang)"

        ax.text(x, y, pair_label,
                ha="center", va="center", fontsize=12,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color+"22", edgecolor=color, lw=1.2))
        ax.text(x, y - 0.07, resource_str,
                ha="center", va="center", fontsize=8, color="#666")

# dividers
for ti in range(1, n_tiers):
    ax.axvline(ti * col_w, color="#ddd", lw=1, ymin=0.05, ymax=0.98)

ax.set_title("CodeNet Language Pair Tiers for Qwen-2.5 Fine-tuning\n"
             "(finetune_plan.md §2.2)", fontsize=14, pad=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_lang_pair_tiers.png")
plt.close(fig)
print("Saved 03_lang_pair_tiers.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Data-driven font size guide (2D: lines × max_line_len)
# ═══════════════════════════════════════════════════════════════════════════════
# Font size logic from image_rendering_thinking.md (adapted)
CANVAS_W = 1220   # usable px width (1280 - 2×24 padding + some extra)
CANVAS_H = 744    # usable px height
CHAR_W_FACTOR = 0.8   # char_width ≈ 0.8 * font_size_px
LINE_H_FACTOR = 1.6   # line_height ≈ 1.6 * font_size_px

def pick_font_size(n_lines, max_chars, max_lines=45):
    if   n_lines <= 15: size_by_lines = 24
    elif n_lines <= 30: size_by_lines = 18
    elif n_lines <= 45: size_by_lines = 14
    else:               size_by_lines = 12  # truncation will apply

    size_by_width = int(CANVAS_W / (max_chars * CHAR_W_FACTOR))
    size_by_width = min(size_by_width, 24)

    chosen = min(size_by_lines, size_by_width)
    chosen = max(chosen, 10)
    truncated = n_lines > max_lines and chosen <= 12
    return chosen, truncated


# Build a summary table per language using percentiles
rows = []
for lang in TARGET_LANGS:
    lc = target_stats[lang]["line_count"]
    ml = target_stats[lang]["max_line_len"]
    for label, n_lines_val, max_chars_val in [
        ("P50 (typical)",    lc["median"], ml["median"]),
        ("P90 (long code)",  lc["p90"],    ml["p90"]),
    ]:
        fs, trunc = pick_font_size(int(n_lines_val), int(max_chars_val))
        rows.append({
            "lang": lang,
            "scenario": label,
            "n_lines": n_lines_val,
            "max_chars": max_chars_val,
            "font_size": fs,
            "truncated": trunc,
        })

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: scatter plot (n_lines vs max_chars, colored by font size, per language)
ax = axes[0]
line_grid = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 80])
char_grid = np.array([20, 30, 40, 50, 60, 70, 80, 100, 120, 150])

LL, CC = np.meshgrid(line_grid, char_grid)
FS = np.array([[pick_font_size(int(l), int(c))[0] for l in line_grid] for c in char_grid])
TR = np.array([[pick_font_size(int(l), int(c))[1] for l in line_grid] for c in char_grid])

cmap = plt.cm.get_cmap("RdYlGn_r", 5)
sc = ax.contourf(LL, CC, FS, levels=[10, 12, 14, 18, 22, 25],
                 cmap=cmap, alpha=0.35)
plt.colorbar(sc, ax=ax, label="Font size (pt)", shrink=0.8)

# Hatch truncation zone
ax.contourf(LL, CC, FS.astype(float), levels=[9.5, 10.5],
            hatches=["///"], colors="none", alpha=0.2)

# Overlay actual language stats
for lang in TARGET_LANGS:
    lc = target_stats[lang]["line_count"]
    ml = target_stats[lang]["max_line_len"]
    color = LANG_COLORS[lang]
    # P50 dot (filled)
    ax.scatter(lc["median"], ml["median"], s=120, color=color, zorder=10,
               edgecolors="white", lw=1.5, label=lang)
    # P90 dot (open)
    ax.scatter(lc["p90"], ml["p90"], s=80, color=color, zorder=10,
               edgecolors=color, lw=1.5, marker="D", facecolors="none")
    # Arrow P50→P90
    ax.annotate("", xy=(lc["p90"], ml["p90"]),
                xytext=(lc["median"], ml["median"]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.6))

ax.set_xlabel("File line count")
ax.set_ylabel("Max line length (chars)")
ax.set_title("Font Size Landscape\n●=P50 ◇=P90, arrow=P50→P90 per language")
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.set_xlim(0, 230)
ax.set_ylim(10, 160)
ax.grid(lw=0.4, alpha=0.4)

# Right: summary table
ax = axes[1]
ax.axis("off")

col_labels = ["Language", "Scenario", "Lines\n(pct)", "MaxLen\n(pct)", "Font(pt)", "Truncate?"]
table_data = []
for r in rows:
    trunc_str = "YES ⚠️" if r["truncated"] else "no"
    table_data.append([
        r["lang"],
        r["scenario"].replace(" (typical)", "\n(typical)").replace(" (long code)", "\n(long code)"),
        f"{r['n_lines']:.0f}",
        f"{r['max_chars']:.0f}",
        str(r["font_size"]),
        trunc_str,
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.6)

# Color rows by language
lang_row_map = {}
for i, r in enumerate(rows):
    row_idx = i + 1  # +1 for header
    base_color = LANG_COLORS[r["lang"]]
    for col in range(len(col_labels)):
        cell = tbl[row_idx, col]
        cell.set_facecolor(base_color + "22")
        if r["truncated"] and col == 5:
            cell.set_facecolor("#FF5722" + "55")
    # Header rows
    if i % 2 == 0:
        tbl[row_idx, 0].set_text_props(fontweight="bold")

# Style header
for col in range(len(col_labels)):
    tbl[0, col].set_facecolor("#37474F")
    tbl[0, col].set_text_props(color="white", fontweight="bold")

ax.set_title("Data-Driven Font Size Recommendations\n(based on CodeNet empirical P50 / P90)", pad=15)

fig.suptitle("Rendering Pipeline Font Size Guide — Selected 7 Languages", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_font_size_guide.png", bbox_inches="tight")
plt.close(fig)
print("Saved 04_font_size_guide.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Combined summary: 2×2 panel for the paper (Appendix / Fig 1 candidate)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_lc  = fig.add_subplot(gs[0, 0])   # line count box
ax_ml  = fig.add_subplot(gs[0, 1])   # max line len box
ax_2d  = fig.add_subplot(gs[1, 0])   # 2D scatter
ax_fs  = fig.add_subplot(gs[1, 1])   # font size bar

# ─ (a) Line count ─
percentile_keys = ["min", "p25", "median", "p75", "p90"]
for i, lang in enumerate(TARGET_LANGS):
    lc = target_stats[lang]["line_count"]
    color = LANG_COLORS[lang]
    ax_lc.barh(i, lc["p75"] - lc["p25"], left=lc["p25"], height=0.55,
               color=color, alpha=0.78, zorder=3)
    ax_lc.plot([lc["median"]] * 2, [i - 0.28, i + 0.28], color="white", lw=2, zorder=4)
    ax_lc.plot([lc["p25"], lc["min"]], [i, i], color=color, lw=1.2, alpha=0.5)
    ax_lc.plot([lc["p75"], lc["p90"]], [i, i], color=color, lw=1.2, alpha=0.5)
    ax_lc.scatter([lc["p95"]], [i], marker="|", s=80, color=color, lw=2, zorder=5)

ax_lc.set_yticks(range(len(TARGET_LANGS)))
ax_lc.set_yticklabels(TARGET_LANGS, fontsize=11)
ax_lc.axvline(45, color="#999", lw=1, ls="--", alpha=0.7)
ax_lc.set_xlabel("Lines of code")
ax_lc.set_title("(a) Line Count  [box=IQR, tick=P95]")
ax_lc.set_xlim(-5, 260)
ax_lc.grid(axis="x", lw=0.3, alpha=0.4)

# ─ (b) Max line length ─
for i, lang in enumerate(TARGET_LANGS):
    ml = target_stats[lang]["max_line_len"]
    color = LANG_COLORS[lang]
    ax_ml.barh(i, ml["p75"] - ml["p25"], left=ml["p25"], height=0.55,
               color=color, alpha=0.78, zorder=3)
    ax_ml.plot([ml["median"]] * 2, [i - 0.28, i + 0.28], color="white", lw=2, zorder=4)
    ax_ml.plot([ml["p25"], ml["min"]], [i, i], color=color, lw=1.2, alpha=0.5)
    ax_ml.plot([ml["p75"], ml["p90"]], [i, i], color=color, lw=1.2, alpha=0.5)
    ax_ml.scatter([ml["p95"]], [i], marker="|", s=80, color=color, lw=2, zorder=5)

ax_ml.set_yticks(range(len(TARGET_LANGS)))
ax_ml.set_yticklabels(TARGET_LANGS, fontsize=11)
ax_ml.axvline(80, color="#999", lw=1, ls="--", alpha=0.7, label="80-char")
ax_ml.set_xlabel("Max line length (chars)")
ax_ml.set_title("(b) Max Line Length  [box=IQR, tick=P95]")
ax_ml.set_xlim(0, 180)
ax_ml.grid(axis="x", lw=0.3, alpha=0.4)

# ─ (c) 2D scatter ─
for lang in TARGET_LANGS:
    lc = target_stats[lang]["line_count"]
    ml = target_stats[lang]["max_line_len"]
    color = LANG_COLORS[lang]
    ax_2d.scatter(lc["median"], ml["median"], s=180, color=color, zorder=10,
                  edgecolors="white", lw=1.5)
    ax_2d.scatter(lc["p90"], ml["p90"], s=80, color=color, zorder=10,
                  edgecolors=color, lw=1.5, marker="D", facecolors="none")
    ax_2d.annotate("", xy=(lc["p90"], ml["p90"]),
                   xytext=(lc["median"], ml["median"]),
                   arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.55))
    ax_2d.text(lc["median"] + 1, ml["median"] + 1, lang, fontsize=8.5,
               color=color, fontweight="bold")

ax_2d.set_xlabel("Lines (P50 / P90)")
ax_2d.set_ylabel("Max line length chars (P50 / P90)")
ax_2d.set_title("(c) Rendering Complexity\n●=P50  ◇=P90")
ax_2d.axvline(45, color="#aaa", lw=1, ls="--", alpha=0.5, label="45-line truncation")
ax_2d.axhline(80, color="#bbb", lw=1, ls=":", alpha=0.5)
ax_2d.legend(fontsize=8)
ax_2d.grid(lw=0.3, alpha=0.4)
ax_2d.set_xlim(0, 220)
ax_2d.set_ylim(25, 135)

# ─ (d) Recommended font sizes (P50 / P90) ─
x = np.arange(len(TARGET_LANGS))
width = 0.35
fs_p50 = []
fs_p90 = []
for lang in TARGET_LANGS:
    lc = target_stats[lang]["line_count"]
    ml = target_stats[lang]["max_line_len"]
    fs_p50.append(pick_font_size(int(lc["median"]), int(ml["median"]))[0])
    fs_p90.append(pick_font_size(int(lc["p90"]),    int(ml["p90"]))[0])

bars1 = ax_fs.bar(x - width/2, fs_p50, width, label="Font at P50 (typical)",
                  color=[LANG_COLORS[l] for l in TARGET_LANGS], alpha=0.85)
bars2 = ax_fs.bar(x + width/2, fs_p90, width, label="Font at P90 (long code)",
                  color=[LANG_COLORS[l] for l in TARGET_LANGS], alpha=0.45,
                  edgecolor=[LANG_COLORS[l] for l in TARGET_LANGS], lw=1.5)

for bar, val in zip(bars1, fs_p50):
    ax_fs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, fs_p90):
    ax_fs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               str(val), ha="center", va="bottom", fontsize=9)

ax_fs.set_xticks(x)
ax_fs.set_xticklabels(TARGET_LANGS, rotation=20, ha="right")
ax_fs.set_ylabel("Recommended font size (pt)")
ax_fs.set_ylim(0, 30)
ax_fs.set_title("(d) Data-Driven Font Size\n(image_rendering_thinking formula)")
ax_fs.legend(fontsize=9)
ax_fs.axhline(12, color="#E53935", lw=1, ls="--", alpha=0.5, label="min font (12pt)")
ax_fs.grid(axis="y", lw=0.3, alpha=0.4)

fig.suptitle("CodeNet Selected Language Stats — Rendering Pipeline Calibration\n"
             "Seed 42, 100 files per language", fontsize=14, fontweight="bold")
fig.savefig(OUT_DIR / "05_combined_summary.png", bbox_inches="tight")
plt.close(fig)
print("Saved 05_combined_summary.png")

print(f"\nAll figures written to {OUT_DIR}/")

# ── Print selection rationale ─────────────────────────────────────────────────
print("\n" + "="*70)
print("SELECTED LANGUAGE PAIRS (from finetune_plan.md §2.2)")
print("="*70)
for tier, pairs in TIERS.items():
    print(f"\n{tier.replace(chr(10), ' ')}:")
    for l1, l2 in pairs:
        lc1 = target_stats[l1]["line_count"]
        lc2 = target_stats[l2]["line_count"]
        f1  = target_stats[l1]["total_files"]
        f2  = target_stats[l2]["total_files"]
        fs1_p50, _ = pick_font_size(int(lc1["median"]), int(target_stats[l1]["max_line_len"]["median"]))
        fs2_p50, _ = pick_font_size(int(lc2["median"]), int(target_stats[l2]["max_line_len"]["median"]))
        print(f"  {l1:8s} ({f1/1e3:7.0f}K files, med={lc1['median']:4.0f}L, font≈{fs1_p50}pt) "
              f"↔ {l2:8s} ({f2/1e3:7.0f}K files, med={lc2['median']:4.0f}L, font≈{fs2_p50}pt)")
