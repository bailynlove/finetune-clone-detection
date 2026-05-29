#!/usr/bin/env python3
"""Extract training loss from HF trainer logs and plot curves for all groups."""
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LOGS = {
    "B1 (text-only)":      "runs/b1_train_a_seed42.log",
    "B1ctrl (black+text)": "runs/b1ctrl_train_a_seed42.log",
    "B2 (image+text)":     "runs/b2_train_a_seed42.log",
    "B3 (image-only)":     "runs/b3_train_a_seed42.log",
}

COLORS = {
    "B1 (text-only)":      "#1f77b4",
    "B1ctrl (black+text)": "#ff7f0e",
    "B2 (image+text)":     "#2ca02c",
    "B3 (image-only)":     "#d62728",
}

DICT_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")

def parse_log(path):
    steps, losses = [], []
    step = 0
    with open(path, errors="replace") as f:
        for raw_line in f:
            for line in raw_line.split("\r"):
                for m in DICT_RE.finditer(line):
                    try:
                        d = json.loads(m.group().replace("'", '"'))
                        if "loss" in d and "eval_loss" not in m.group():
                            step += 1
                            steps.append(float(d.get("epoch", step)))
                            losses.append(float(d["loss"]))
                    except Exception:
                        pass
    return steps, losses

root = Path("/data1/clone-test")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: full scale ──────────────────────────────────────────────────────────
ax = axes[0]
for label, rel_path in LOGS.items():
    path = root / rel_path
    if not path.exists():
        print(f"Missing: {path}")
        continue
    epochs, losses = parse_log(path)
    if not epochs:
        print(f"No data: {label}")
        continue
    ax.plot(epochs, losses, label=label, color=COLORS[label], linewidth=1.5, alpha=0.85)
    print(f"{label}: {len(epochs)} points, final_loss={losses[-1]:.6f}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Curves (full scale)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0)

# ── Right: zoomed (loss < 0.5 to skip early spike) ────────────────────────────
ax2 = axes[1]
for label, rel_path in LOGS.items():
    path = root / rel_path
    if not path.exists():
        continue
    epochs, losses = parse_log(path)
    if not epochs:
        continue
    # filter to first 3 epochs and loss < 0.3 for zoom
    ep_z = [e for e, l in zip(epochs, losses) if l < 0.3]
    lo_z = [l for l in losses if l < 0.3]
    ax2.plot(ep_z, lo_z, label=label, color=COLORS[label], linewidth=1.5, alpha=0.85)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Training Loss")
ax2.set_title("Training Loss Curves (loss < 0.3 zoom)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0)

plt.tight_layout()
out = root / "figures/loss_curves.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
