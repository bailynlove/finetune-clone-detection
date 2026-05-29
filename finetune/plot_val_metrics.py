#!/usr/bin/env python3
"""Plot per-epoch val accuracy and F1 for B1 and B2."""
import re, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/data1/clone-test")

DICT_RE = re.compile(r"\{[^{}]*'eval_loss'[^{}]*\}")

def parse_eval_log(path):
    results = []
    with open(path, errors="replace") as f:
        for raw in f:
            for line in raw.split("\r"):
                for m in DICT_RE.finditer(line):
                    try:
                        d = json.loads(m.group().replace("'", '"'))
                        if "eval_loss" in d and "epoch" in d:
                            results.append({
                                "epoch": float(d["epoch"]),
                                "accuracy": float(d.get("eval_mean_token_accuracy", 0)),
                                "eval_loss": float(d["eval_loss"]),
                            })
                    except Exception:
                        pass
    return results

def load_f1_from_results(pattern_glob):
    """Load F1 from eval.py output JSONs matching a glob pattern."""
    results = {}
    for p in sorted(ROOT.glob(pattern_glob)):
        try:
            d = json.loads(p.read_text())
            cp = d.get("checkpoint", "")
            f1 = d.get("f1")
            acc = d.get("accuracy")
            if f1 is not None and cp:
                results[cp] = {"f1": f1, "accuracy": acc}
        except Exception:
            pass
    return results

b1_eval = parse_eval_log(ROOT / "runs/b1_train_a_seed42.log")
b2_eval = parse_eval_log(ROOT / "runs/b2_train_a_seed42.log")

print("B1 per-epoch val metrics from trainer:")
for r in b1_eval:
    print(f"  epoch {r['epoch']:.0f}: acc={r['accuracy']:.4f}  loss={r['eval_loss']:.5f}")

print("B2 per-epoch val metrics from trainer:")
for r in b2_eval:
    print(f"  epoch {r['epoch']:.0f}: acc={r['accuracy']:.4f}  loss={r['eval_loss']:.5f}")

# Load any val-set F1 results (from eval.py runs on val split)
b1_val_f1 = load_f1_from_results("results/b1_val_ep*.json")
b2_val_f1 = load_f1_from_results("results/b2_val_ep*.json")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

groups = [
    ("B1 (text-only)", b1_eval, "#1f77b4", b1_val_f1),
    ("B2 (image+text)", b2_eval, "#2ca02c", b2_val_f1),
]

for ax_idx, (label, evals, color, f1_data) in enumerate(groups):
    epochs  = [r["epoch"] for r in evals]
    accs    = [r["accuracy"] * 100 for r in evals]

    axes[0].plot(epochs, accs, "o-", color=color, label=label, linewidth=2, markersize=7)
    for e, a in zip(epochs, accs):
        axes[0].annotate(f"{a:.2f}%", (e, a), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=color)

    # F1 from eval.py runs (if available)
    if f1_data:
        ep_f1 = []
        for cp, vals in f1_data.items():
            # infer epoch from checkpoint name
            try:
                step = int(re.search(r"checkpoint-(\d+)", cp).group(1))
                ep = step / (len(b1_eval[0]) if b1_eval else 1250)
                # steps/epoch = 20000/16 = 1250
                ep = step / 1250
                ep_f1.append((ep, vals["f1"] * 100))
            except Exception:
                pass
        if ep_f1:
            ep_f1.sort()
            axes[1].plot([e for e, f in ep_f1], [f for e, f in ep_f1],
                         "s--", color=color, label=label, linewidth=2, markersize=7)

for ax in axes[:1 if not any(b1_val_f1 or b2_val_f1 for *_, f1 in groups if f1) else 2]:
    pass

# Val accuracy subplot
ax = axes[0]
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val Accuracy (%)", fontsize=12)
ax.set_title("Validation Accuracy per Epoch\n(eval_mean_token_accuracy, ≈ binary Yes/No accuracy)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 2, 3])
ax.set_xlim(0.7, 3.3)
y_min = min(min(r["accuracy"] for r in b1_eval), min(r["accuracy"] for r in b2_eval)) * 100
ax.set_ylim(max(97, y_min - 0.3), 100.2)

# Val F1 subplot — use trainer accuracy as proxy if no dedicated F1 runs
ax2 = axes[1]
# Load actual val F1 if available, else use trainer accuracy as F1 proxy (balanced dataset)
for label, evals, color, f1_data in groups:
    epochs = [r["epoch"] for r in evals]
    # Use actual F1 if available, else accuracy as proxy
    f1_vals = []
    for ep, r in zip(epochs, evals):
        f1_vals.append(r["accuracy"] * 100)  # proxy; will overlay actual if available

    ax2.plot(epochs, f1_vals, "o-", color=color, label=f"{label} (acc proxy)",
             linewidth=2, markersize=7, linestyle="--", alpha=0.6)
    for e, f in zip(epochs, f1_vals):
        ax2.annotate(f"{f:.2f}%", (e, f), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=color)

ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Val F1 (%)", fontsize=12)
ax2.set_title("Validation F1 per Epoch\n(using trainer accuracy as proxy — see note)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([1, 2, 3])
ax2.set_xlim(0.7, 3.3)
ax2.set_ylim(max(97, y_min - 0.3), 100.2)

fig.text(0.5, -0.02,
         "Note: trainer logs eval_mean_token_accuracy (binary accuracy on masked Yes/No token).\n"
         "For balanced labels, F1 ≈ accuracy. Exact per-epoch F1 requires inference on each checkpoint.",
         ha="center", fontsize=8, color="gray", wrap=True)

plt.tight_layout()
out = ROOT / "figures/val_metrics_b1_b2.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
