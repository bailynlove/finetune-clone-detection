#!/usr/bin/env python3
"""
For each language in Project_CodeNet, randomly sample up to 100 code files
and compute:
  - line count distribution (min, p25, median, p75, p90, p95, max, mean)
  - max line length distribution (same percentiles)
"""

import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/data1/clone-test/Project_CodeNet/data")
SAMPLE_N = 100
SEED = 42

random.seed(SEED)

# Collect all files per language
lang_files = defaultdict(list)
for problem_dir in DATA_DIR.iterdir():
    if not problem_dir.is_dir():
        continue
    for lang_dir in problem_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for f in lang_dir.iterdir():
            if f.is_file():
                lang_files[lang].append(f)

print(f"Found {len(lang_files)} languages\n")

def percentile(data, p):
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

def stats(data):
    if not data:
        return {}
    return {
        "n":      len(data),
        "min":    min(data),
        "p25":    round(percentile(data, 25), 1),
        "median": round(percentile(data, 50), 1),
        "mean":   round(statistics.mean(data), 1),
        "p75":    round(percentile(data, 75), 1),
        "p90":    round(percentile(data, 90), 1),
        "p95":    round(percentile(data, 95), 1),
        "max":    max(data),
    }

results = {}

langs_sorted = sorted(lang_files.keys(), key=lambda l: -len(lang_files[l]))
for lang in langs_sorted:
    files = lang_files[lang]
    sample = random.sample(files, min(SAMPLE_N, len(files)))

    line_counts = []
    max_line_lens = []

    for fpath in sample:
        try:
            content = fpath.read_text(errors="replace")
            lines = content.splitlines()
            line_counts.append(len(lines))
            max_line_lens.append(max((len(l) for l in lines), default=0))
        except Exception:
            pass

    results[lang] = {
        "total_files": len(files),
        "sampled":     len(line_counts),
        "line_count":  stats(line_counts),
        "max_line_len": stats(max_line_lens),
    }

# Print results table
HDR = f"{'Language':<20} {'Files':>8} {'Samp':>5} | {'Lines':^45} | {'MaxLineLen':^45}"
SUB = f"{'':20} {'':8} {'':5} | {'min':>5} {'p25':>6} {'med':>6} {'mean':>7} {'p75':>6} {'p90':>6} {'p95':>6} {'max':>7} | {'min':>5} {'p25':>6} {'med':>6} {'mean':>7} {'p75':>6} {'p90':>6} {'p95':>6} {'max':>7}"

print(HDR)
print(SUB)
print("-" * len(SUB))

for lang, r in results.items():
    lc = r["line_count"]
    ml = r["max_line_len"]
    if not lc:
        continue
    print(
        f"{lang:<20} {r['total_files']:>8} {r['sampled']:>5} | "
        f"{lc['min']:>5} {lc['p25']:>6} {lc['median']:>6} {lc['mean']:>7} {lc['p75']:>6} {lc['p90']:>6} {lc['p95']:>6} {lc['max']:>7} | "
        f"{ml['min']:>5} {ml['p25']:>6} {ml['median']:>6} {ml['mean']:>7} {ml['p75']:>6} {ml['p90']:>6} {ml['p95']:>6} {ml['max']:>7}"
    )

# Also dump as JSON for further use
import json
out_path = Path("/data1/clone-test/codenet_stats.json")
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nFull stats written to {out_path}")
