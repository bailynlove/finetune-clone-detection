#!/usr/bin/env python3
"""Minimal eval that saves per-example predictions (subset eval for diagnosis)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from eval import load_model, run_inference, load_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["b1","b1ctrl","b2","b3"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base_model", default="/data1/models/Qwen3.5-4B")
    ap.add_argument("--test_files", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt_style", default="default", choices=["default","iankappa"])
    args = ap.parse_args()

    records = load_jsonl(args.test_files)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"Evaluating {len(records)} examples mode={args.mode} no_lora={args.no_lora}")
    model, processor = load_model(args.checkpoint, args.base_model, no_lora=args.no_lora)
    model.eval()
    preds, golds = run_inference(model, processor, records, args.mode, args.batch_size,
                                  prompt_style=args.prompt_style)
    out = []
    for r, p, g in zip(records, preds, golds):
        out.append({
            "pair_id":  r.get("pair_id", ""),
            "label":    g,
            "pred":     p,
            "neg_type": r.get("neg_type"),
            "lang_pair": r.get("lang_pair", ""),
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"Saved {len(out)} predictions → {args.output}")
