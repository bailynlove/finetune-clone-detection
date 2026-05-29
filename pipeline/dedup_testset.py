#!/usr/bin/env python3
"""
Deduplicate test sets against training set using character-level Jaccard similarity.

For each code in test_sd/test_dd, compute max Jaccard similarity against all
training codes. Pairs where either code has max_sim > threshold are flagged as
"too similar" and excluded.

Usage:
    python pipeline/dedup_testset.py [--threshold 0.7] [--lang_pair python_java]
"""
import re
import ast
import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

CODENET = Path("/data1/clone-test/Project_CodeNet")
SPLITS  = Path("/data1/clone-test/dataset/splits")
FINETUNE = Path("/data1/clone-test/dataset/finetune_data")

# ── Code loading & normalization ───────────────────────────────────────────────

def load_code(rel_path: str) -> str:
    p = CODENET / rel_path
    try:
        return p.read_text(errors="replace")
    except Exception:
        return ""

def normalize(code: str) -> str:
    """Strip comments, collapse whitespace, lowercase."""
    code = re.sub(r'#[^\n]*', '', code)           # Python comments
    code = re.sub(r'//[^\n]*', '', code)           # C++/Java line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip().lower()
    return code

def shingles(text: str, k: int = 5) -> frozenset:
    return frozenset(text[i:i+k] for i in range(max(0, len(text)-k+1)))

def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ── MinHash approximation ─────────────────────────────────────────────────────

N_PERM = 128
_rng = np.random.RandomState(42)
_A = _rng.randint(1, 2**31-1, N_PERM, dtype=np.int64)
_B = _rng.randint(0, 2**31-1, N_PERM, dtype=np.int64)
_P = np.int64(2**31 - 1)

def minhash(shin: frozenset) -> np.ndarray:
    sig = np.full(N_PERM, _P, dtype=np.int64)
    for s in shin:
        h = hash(s) % _P
        vals = (_A * h + _B) % _P
        sig = np.minimum(sig, vals)
    return sig

def approx_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))

# ── LSH buckets ───────────────────────────────────────────────────────────────

def build_lsh(sigs, n_bands=16):
    """Return dict: band_key -> list of indices with that key."""
    rows = N_PERM // n_bands
    buckets = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(n_bands):
            key = (b, tuple(sig[b*rows:(b+1)*rows]))
            buckets[key].append(idx)
    return buckets, rows

def query_lsh(sig, buckets, rows, n_bands=16):
    """Return set of candidate indices (share at least one band)."""
    candidates = set()
    for b in range(n_bands):
        key = (b, tuple(sig[b*rows:(b+1)*rows]))
        candidates.update(buckets.get(key, []))
    return candidates

# ── Main ──────────────────────────────────────────────────────────────────────

def load_records(split_path: Path):
    records = []
    with open(split_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                c1 = r['code1']
                c2 = r['code2']
                if isinstance(c1, str): c1 = ast.literal_eval(c1)
                if isinstance(c2, str): c2 = ast.literal_eval(c2)
                records.append((r['pair_id'], r['label'], c1, c2))
            except Exception:
                pass
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang_pair", default="python_java")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Max Jaccard similarity allowed; pairs above this are removed")
    ap.add_argument("--n_bands", type=int, default=16,
                    help="LSH bands (fewer = higher recall, lower precision)")
    ap.add_argument("--analyze", action="store_true",
                    help="Only show similarity distribution, don't write output")
    ap.add_argument("--sample_n", type=int, default=0,
                    help="Sample N pairs for quick analysis (0 = all)")
    args = ap.parse_args()

    lang = args.lang_pair
    base = SPLITS / lang
    
    print(f"Loading train_a for {lang}...")
    train_recs = load_records(base / "train_a.jsonl")
    print(f"  {len(train_recs)} train pairs")

    # Load and normalize all training codes
    print("Loading & normalizing training codes...")
    train_codes_raw = []
    for pid, label, c1, c2 in train_recs:
        train_codes_raw.append((c1['rel_path'], load_code(c1['rel_path'])))
        train_codes_raw.append((c2['rel_path'], load_code(c2['rel_path'])))

    print("Computing training shingles + MinHash...")
    train_norm  = [normalize(c) for _, c in train_codes_raw]
    train_shin  = [shingles(n) for n in train_norm]
    train_sigs  = [minhash(s) for s in train_shin]
    print(f"  {len(train_sigs)} training codes indexed")

    buckets, rows = build_lsh(train_sigs, n_bands=args.n_bands)
    print(f"  LSH: {args.n_bands} bands, {rows} rows/band, {len(buckets)} buckets")

    # Process test sets
    for split in ["test_sd", "test_dd"]:
        split_path = base / f"{split}.jsonl"
        if not split_path.exists():
            continue

        print(f"\nProcessing {split}...")
        test_recs = load_records(split_path)
        if args.sample_n:
            random.seed(42)
            test_recs = random.sample(test_recs, min(args.sample_n, len(test_recs)))

        kept, removed = [], []
        sims_all = []

        for i, (pid, label, c1, c2) in enumerate(test_recs):
            if i % 100 == 0:
                print(f"  {i}/{len(test_recs)}...", end="\r", flush=True)

            max_sim = 0.0
            pair_detail = []

            for code_meta in [c1, c2]:
                raw  = load_code(code_meta['rel_path'])
                norm = normalize(raw)
                shin = shingles(norm)
                sig  = minhash(shin)

                # LSH candidate lookup
                cands = query_lsh(sig, buckets, rows, args.n_bands)
                
                # Exact Jaccard on candidates only
                best = 0.0
                best_idx = -1
                for idx in cands:
                    j = jaccard(shin, train_shin[idx])
                    if j > best:
                        best = j
                        best_idx = idx
                max_sim = max(max_sim, best)
                pair_detail.append((code_meta['rel_path'], best, 
                                    train_codes_raw[best_idx][0] if best_idx >= 0 else ""))

            sims_all.append(max_sim)
            if max_sim > args.threshold:
                removed.append((pid, label, max_sim, pair_detail))
            else:
                kept.append(pid)

        sims_all = np.array(sims_all)
        print(f"\n  Total: {len(test_recs)}  Kept: {len(kept)}  Removed: {len(removed)}")
        print(f"  Similarity stats: min={sims_all.min():.3f} "
              f"mean={sims_all.mean():.3f} "
              f"median={np.median(sims_all):.3f} "
              f"max={sims_all.max():.3f} "
              f"p90={np.percentile(sims_all, 90):.3f} "
              f"p95={np.percentile(sims_all, 95):.3f}")

        # Quantile breakdown
        for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
            n_above = int((sims_all > thr).sum())
            print(f"  sim > {thr}: {n_above} pairs ({100*n_above/len(test_recs):.1f}%)")

        if removed:
            print(f"\n  Examples of removed pairs (threshold={args.threshold}):")
            for pid, label, sim, detail in removed[:5]:
                print(f"    {pid}  sim={sim:.3f}")
                for path, s, train_path in detail:
                    print(f"      test:  {path}  (best={s:.3f})")
                    if train_path:
                        print(f"      train: {train_path}")

        if not args.analyze:
            # Write filtered finetune JSONL for each mode
            for mode in ["b1", "b1ctrl", "b2", "b3"]:
                src = FINETUNE / mode / split / f"{lang}.jsonl"
                dst = FINETUNE / mode / f"{split}_dedup" / f"{lang}.jsonl"
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                kept_set = set(kept)
                n_in, n_out = 0, 0
                with open(src) as fin, open(dst, "w") as fout:
                    for line in fin:
                        try:
                            r = json.loads(line)
                            if r.get("pair_id") in kept_set:
                                fout.write(line)
                                n_out += 1
                            n_in += 1
                        except Exception:
                            pass
                print(f"  [{mode}] {split}: {n_in} → {n_out} written to {dst}")

if __name__ == "__main__":
    main()
