#!/usr/bin/env python3
"""
Two-stage test-set deduplication against training set.

Stage 1: Exact submission ID match (hard exclusion — same code file in both train & test).
Stage 2: Code similarity > threshold (MinHash Jaccard approximation).

Writes deduplicated JSONL files for all modes (b1/b1ctrl/b2/b3).
"""
import re, ast, json, argparse, numpy as np
from pathlib import Path
from collections import defaultdict

CODENET = Path("/data1/clone-test/Project_CodeNet")
SPLITS  = Path("/data1/clone-test/dataset/splits")
FINETUNE = Path("/data1/clone-test/dataset/finetune_data")

# ── Code utils ────────────────────────────────────────────────────────────────

def load_code(rel_path):
    try:
        return (CODENET / rel_path).read_text(errors="replace")
    except Exception:
        return ""

def normalize(code):
    code = re.sub(r'#[^\n]*', '', code)
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', code).strip().lower()

def shingles(text, k=5):
    return frozenset(text[i:i+k] for i in range(max(0, len(text)-k+1)))

def jaccard(a, b):
    if not a and not b: return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0

# ── MinHash ───────────────────────────────────────────────────────────────────

N_PERM = 128
_rng = np.random.RandomState(42)
_A = _rng.randint(1, 2**31-1, N_PERM, dtype=np.int64)
_B = _rng.randint(0, 2**31-1, N_PERM, dtype=np.int64)
_P = np.int64(2**31 - 1)

def minhash(shin):
    sig = np.full(N_PERM, _P, dtype=np.int64)
    for s in shin:
        h = hash(s) % _P
        sig = np.minimum(sig, (_A * h + _B) % _P)
    return sig

def build_lsh(sigs, n_bands=16):
    rows = N_PERM // n_bands
    buckets = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(n_bands):
            buckets[(b, tuple(sig[b*rows:(b+1)*rows]))].append(idx)
    return buckets, rows

def query_lsh(sig, buckets, rows, n_bands=16):
    candidates = set()
    for b in range(n_bands):
        candidates.update(buckets.get((b, tuple(sig[b*rows:(b+1)*rows])), []))
    return candidates

# ── Record loading ─────────────────────────────────────────────────────────────

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                c1 = ast.literal_eval(r['code1']) if isinstance(r['code1'], str) else r['code1']
                c2 = ast.literal_eval(r['code2']) if isinstance(r['code2'], str) else r['code2']
                records.append({
                    'pair_id': r['pair_id'],
                    'label':   r['label'],
                    'c1':      c1,
                    'c2':      c2,
                })
            except Exception:
                pass
    return records

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang_pair", default="python_java")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--n_bands",   type=int,   default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    lang = args.lang_pair
    base = SPLITS / lang

    # ── Load training data ────────────────────────────────────────────────────
    print(f"Loading train_a for {lang}...")
    train_recs = load_records(base / "train_a.jsonl")

    train_sub_ids = set()
    train_codes_meta = []  # (rel_path, normalized_code)
    for r in train_recs:
        for c in [r['c1'], r['c2']]:
            train_sub_ids.add(c['submission_id'])
            norm = normalize(load_code(c['rel_path']))
            train_codes_meta.append((c['rel_path'], norm))

    print(f"  {len(train_recs)} pairs, {len(train_sub_ids)} unique submissions")

    # MinHash index for similarity dedup
    print("Building MinHash index...")
    train_shin = [shingles(n) for _, n in train_codes_meta]
    train_sigs = [minhash(s) for s in train_shin]
    buckets, rows = build_lsh(train_sigs, args.n_bands)
    print(f"  {len(train_shin)} codes indexed, {len(buckets)} LSH buckets")

    # ── Process each test split ────────────────────────────────────────────────
    for split in ["test_sd", "test_dd"]:
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue
        print(f"\n{'='*60}")
        print(f"Processing {split} ({lang})")
        test_recs = load_records(path)

        kept_ids   = []
        removed_s1 = []   # stage 1: submission ID match
        removed_s2 = []   # stage 2: similarity threshold

        for i, r in enumerate(test_recs):
            if i % 200 == 0:
                print(f"  {i}/{len(test_recs)}...", end="\r")

            # Stage 1: exact submission ID check
            has_exact = any(c['submission_id'] in train_sub_ids
                           for c in [r['c1'], r['c2']])
            if has_exact:
                removed_s1.append(r['pair_id'])
                continue

            # Stage 2: code similarity
            max_sim = 0.0
            for c in [r['c1'], r['c2']]:
                norm = normalize(load_code(c['rel_path']))
                shin = shingles(norm)
                sig  = minhash(shin)
                cands = query_lsh(sig, buckets, rows, args.n_bands)
                for idx in cands:
                    j = jaccard(shin, train_shin[idx])
                    if j > max_sim:
                        max_sim = j

            if max_sim > args.threshold:
                removed_s2.append((r['pair_id'], max_sim))
            else:
                kept_ids.append(r['pair_id'])

        print(f"\n  Results:")
        print(f"    Total:             {len(test_recs)}")
        print(f"    Stage 1 removed (exact submission): {len(removed_s1)} ({100*len(removed_s1)/len(test_recs):.1f}%)")
        print(f"    Stage 2 removed (sim>{args.threshold}):    {len(removed_s2)} ({100*len(removed_s2)/len(test_recs):.1f}%)")
        print(f"    Kept:              {len(kept_ids)} ({100*len(kept_ids)/len(test_recs):.1f}%)")

        if removed_s2:
            print(f"\n  Stage 2 examples:")
            for pid, sim in removed_s2[:5]:
                print(f"    {pid}  sim={sim:.3f}")

        if not args.dry_run:
            kept_set = set(kept_ids)
            for mode in ["b1", "b1ctrl", "b2", "b3"]:
                src = FINETUNE / mode / split / f"{lang}.jsonl"
                dst = FINETUNE / mode / f"{split}_dedup" / f"{lang}.jsonl"
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                n_out = 0
                with open(src) as fin, open(dst, "w") as fout:
                    for line in fin:
                        try:
                            rec = json.loads(line)
                            if rec.get("pair_id") in kept_set:
                                fout.write(line)
                                n_out += 1
                        except Exception:
                            pass
                print(f"  [{mode}] {dst.name}: {n_out} pairs written")

            # Also write the raw splits dedup
            dst_raw = base / f"{split}_dedup.jsonl"
            n_out = 0
            with open(path) as fin, open(dst_raw, "w") as fout:
                for line in fin:
                    try:
                        rec = json.loads(line)
                        if rec.get("pair_id") in kept_set:
                            fout.write(line)
                            n_out += 1
                    except Exception:
                        pass
            print(f"  Raw split → {dst_raw.name}: {n_out} pairs")

if __name__ == "__main__":
    main()
