#!/usr/bin/env python3
"""
01_build_dataset.py
Builds CodeNet cross-language code clone detection datasets.

Key design decisions:
 - NO code truncation: only sample submissions whose full code fits in the render canvas
 - Length-stratified sampling: 4 equal-size buckets on max(n_lines_1, n_lines_2)
 - Efficient I/O: per (problem, lang) pre-select the 10 shortest (by code_size) accepted
   submissions, then read only those files to get exact line counts (~100-200K reads total)
"""

import json, csv, random, os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

CODENET_DIR = Path("/data1/clone-test/Project_CodeNet")
DATASET_DIR = Path("/data1/clone-test/dataset")
SPLITS_DIR  = DATASET_DIR / "splits"
SEED        = 42

# --- Rendering limits (canvas 1280×800, MIN_FONT=12pt, WRAP_WIDTH=56) ---
# Long lines are wrapped at render time; MAX_CHARS applies to original source.
# MIN_LINES filters out trivially short snippets (too few lines for meaningful clone judgment).
MIN_LINES  = 4
MAX_LINES  = 74
MAX_CHARS  = 148
CANDIDATES_PER_LANG_PROB = 10   # read this many (smallest code_size) per (prob, lang)
MAX_SUBS   = 3                  # keep this many passing subs per (prob, lang)

TARGET_LANGS = ["Python", "Java", "C++", "Ruby", "Rust", "Kotlin", "Scala"]

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

TRAIN_A_PAIRS = {("Python", "Java")}
TRAIN_B_PAIRS = {("Python", "Java"), ("Python", "C++"), ("Rust", "Java")}
WEEK1_PAIRS   = {("Python", "Java"), ("Rust", "Java"), ("Rust", "Python"), ("Rust", "Ruby")}

N_TRAIN_A  = 20_000
N_TRAIN_B  = 10_000
N_VAL      =  3_000
N_TEST_SD  =  1_000
N_TEST_DD  =  1_000
N_WEEK1    =    200   # 100 SD + 100 DD

LENGTH_BUCKETS = [(1, 15), (16, 30), (31, 50), (51, MAX_LINES)]


# ── Module-level functions (must be picklable) ────────────────────────────────

def parse_one_csv(args):
    """Parse one problem CSV. Returns (pid, {lang: [(sid, ext, code_size), ...]})."""
    pid, csv_path = args
    result = defaultdict(list)
    try:
        with open(csv_path, newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "Accepted" and row.get("language") in TARGET_LANGS:
                    try:
                        cs = int(row.get("code_size") or 0)
                    except ValueError:
                        cs = 0
                    result[row["language"]].append(
                        (row["submission_id"], row["filename_ext"], cs)
                    )
    except Exception:
        pass
    return pid, dict(result)


def read_line_stats(args):
    """Read a code file, return (pid, lang, sid, ext, n_lines, max_chars) or None."""
    pid, lang, sid, ext = args
    path = CODENET_DIR / "data" / pid / lang / f"{sid}.{ext}"
    try:
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        if not lines:
            return None
        n = len(lines)
        m = max(len(l.expandtabs(4)) for l in lines)
        return (pid, lang, sid, ext, n, m)
    except Exception:
        return None


# ── Pair construction helpers ─────────────────────────────────────────────────

def bucket_of(n_lines):
    for i, (lo, hi) in enumerate(LENGTH_BUCKETS):
        if lo <= n_lines <= hi:
            return i
    return len(LENGTH_BUCKETS) - 1


def stratified_sample(items, n_target, key_fn, n_buckets, rng):
    by_bucket = defaultdict(list)
    for item in items:
        by_bucket[key_fn(item)].append(item)

    per_bucket = max(1, n_target // n_buckets)
    sampled, leftovers = [], []
    for b in range(n_buckets):
        pool = list(by_bucket[b])
        rng.shuffle(pool)
        sampled.extend(pool[:per_bucket])
        leftovers.extend(pool[per_bucket:])

    if len(sampled) < n_target:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:n_target - len(sampled)])

    rng.shuffle(sampled)
    return sampled[:n_target]


def generate_pairs(l1, l2, subs, pool_probs, n_target, rng, tag):
    """Generate n_target//2 pos + n_target//2 neg pairs, length-stratified."""
    overlap = [p for p in pool_probs if p in subs[l1] and p in subs[l2]]
    if not overlap:
        return []

    def pick(lang, pid):
        pool = subs[lang][pid]
        return rng.sample(pool, min(MAX_SUBS, len(pool)))

    pl1 = {p: pick(l1, p) for p in overlap}
    pl2 = {p: pick(l2, p) for p in overlap}

    # All possible positive pairs: (s1, p, s2, p, max_len)
    all_pos = []
    for p in overlap:
        for s1 in pl1[p]:
            for s2 in pl2[p]:
                all_pos.append((s1, p, s2, p, max(s1[2], s2[2])))

    if not all_pos:
        return []

    n_pos = n_target // 2
    sampled_pos = stratified_sample(
        all_pos, n_pos, key_fn=lambda x: bucket_of(x[4]),
        n_buckets=len(LENGTH_BUCKETS), rng=rng
    )

    # Build negative pool
    all_l1 = [(s, p) for p in overlap for s in pl1[p]]
    all_l2 = [(s, p) for p in overlap for s in pl2[p]]
    seen_neg, neg_pool = set(), []
    for _ in range(n_pos * 30):
        if len(neg_pool) >= n_pos * 3:
            break
        s1, p1 = rng.choice(all_l1)
        s2, p2 = rng.choice(all_l2)
        if p1 != p2:
            key = (s1[0], s2[0])
            if key not in seen_neg:
                seen_neg.add(key)
                neg_pool.append((s1, p1, s2, p2, max(s1[2], s2[2])))

    sampled_neg = stratified_sample(
        neg_pool, n_pos, key_fn=lambda x: bucket_of(x[4]),
        n_buckets=len(LENGTH_BUCKETS), rng=rng
    )

    def mk(label, s1, p1, s2, p2, idx):
        sid1, ext1, nl1, mc1 = s1
        sid2, ext2, nl2, mc2 = s2
        kind = "pos" if label == 1 else "neg"
        return {
            "pair_id": f"{tag}_{kind}_{idx:06d}",
            "label": label,
            "code1": {
                "lang": l1, "problem_id": p1, "submission_id": sid1,
                "n_lines": nl1, "max_chars": mc1,
                "rel_path": f"data/{p1}/{l1}/{sid1}.{ext1}",
                "image_rel_path": f"images/{p1}/{l1}/{sid1}.png",
            },
            "code2": {
                "lang": l2, "problem_id": p2, "submission_id": sid2,
                "n_lines": nl2, "max_chars": mc2,
                "rel_path": f"data/{p2}/{l2}/{sid2}.{ext2}",
                "image_rel_path": f"images/{p2}/{l2}/{sid2}.png",
            },
        }

    pairs = (
        [mk(1, s1, p1, s2, p2, i) for i, (s1, p1, s2, p2, _) in enumerate(sampled_pos)] +
        [mk(0, s1, p1, s2, p2, i) for i, (s1, p1, s2, p2, _) in enumerate(sampled_neg)]
    )
    rng.shuffle(pairs)
    return pairs


def save_jsonl(pairs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return len(pairs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Rendering limits: MAX_LINES={MAX_LINES}, MAX_CHARS={MAX_CHARS}")
    print(f"Candidates per (prob,lang): {CANDIDATES_PER_LANG_PROB}, kept after filter: {MAX_SUBS}")
    print(f"Length buckets: {LENGTH_BUCKETS}\n")

    # ── Step 1: Parse metadata CSVs ───────────────────────────────────────────
    print("Step 1: Parsing metadata CSVs in parallel...")
    meta_dir = CODENET_DIR / "metadata"
    all_csv  = sorted(meta_dir.glob("*.csv"))

    # raw_subs[lang][pid] = [(sid, ext, code_size), ...]
    raw_subs = {lang: defaultdict(list) for lang in TARGET_LANGS}

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(parse_one_csv, (p.stem, str(p))): p.stem for p in all_csv}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing CSVs"):
            pid, result = fut.result()
            for lang, subs_list in result.items():
                raw_subs[lang][pid] = subs_list

    for lang in TARGET_LANGS:
        total = sum(len(v) for v in raw_subs[lang].values())
        print(f"  {lang}: {len(raw_subs[lang])} problems, {total} accepted submissions")

    # ── Step 2: Pre-select candidates & read exact line counts ────────────────
    # Per (problem, lang): keep the CANDIDATES_PER_LANG_PROB with smallest code_size
    # (shorter bytes → more likely to fit in line budget → fewer wasted reads)
    print(f"\nStep 2: Reading {CANDIDATES_PER_LANG_PROB} smallest submissions per (prob,lang)...")

    tasks = []
    for lang in TARGET_LANGS:
        for pid, sub_list in raw_subs[lang].items():
            # Sort ascending by code_size, take top CANDIDATES_PER_LANG_PROB
            candidates = sorted(sub_list, key=lambda x: x[2])[:CANDIDATES_PER_LANG_PROB]
            for sid, ext, _cs in candidates:
                tasks.append((pid, lang, sid, ext))

    print(f"  Tasks: {len(tasks):,} file reads")

    # subs[lang][pid] = [(sid, ext, n_lines, max_chars), ...]
    subs = {lang: defaultdict(list) for lang in TARGET_LANGS}
    passed = 0

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(read_line_stats, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reading files"):
            res = fut.result()
            if res is None:
                continue
            pid, lang, sid, ext, n_lines, max_chars = res
            if MIN_LINES <= n_lines <= MAX_LINES and max_chars <= MAX_CHARS:
                subs[lang][pid].append((sid, ext, n_lines, max_chars))
                passed += 1

    print(f"  Passed filter: {passed:,} / {len(tasks):,}")
    for lang in TARGET_LANGS:
        probs  = len(subs[lang])
        total  = sum(len(v) for v in subs[lang].values())
        print(f"  {lang}: {probs} problems, {total} qualifying submissions")

    # ── Step 3: Problem pool split ────────────────────────────────────────────
    print("\nStep 3: Problem pool split (85% train, 5% val, 10% test/DD)...")
    all_pids = sorted(p.stem for p in meta_dir.glob("*.csv"))
    rng_split = random.Random(SEED)
    shuffled  = list(all_pids)
    rng_split.shuffle(shuffled)

    n_total = len(shuffled)
    n_test  = int(n_total * 0.10)
    n_val   = int(n_total * 0.05)

    test_pool  = set(shuffled[:n_test])
    val_pool   = set(shuffled[n_test:n_test + n_val])
    train_pool = set(shuffled[n_test + n_val:])

    print(f"  train={len(train_pool)}, val={len(val_pool)}, test(DD)={len(test_pool)}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "problem_splits.json").write_text(json.dumps({
        "train_pool": sorted(train_pool),
        "val_pool":   sorted(val_pool),
        "test_pool":  sorted(test_pool),
    }, indent=2))

    # ── Step 4: Generate pairs ────────────────────────────────────────────────
    print("\nStep 4: Generating pairs...")
    summary = {}

    for l1, l2 in LANG_PAIRS:
        safe = f"{l1.lower().replace('+','p')}_{l2.lower()}"
        out_dir = SPLITS_DIR / safe
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{l1} ↔ {l2}]")

        def ov(pool):
            return sum(1 for p in pool if p in subs[l1] and p in subs[l2])
        print(f"    Overlap: train={ov(train_pool)}, val={ov(val_pool)}, test(DD)={ov(test_pool)}")

        ps = {"l1": l1, "l2": l2}

        pairs = generate_pairs(l1, l2, subs, list(train_pool), N_TEST_SD,
                               random.Random(SEED+1), f"{safe}_test_sd")
        ps["test_sd"] = save_jsonl(pairs, out_dir / "test_sd.jsonl")
        print(f"    test_sd:  {ps['test_sd']}")

        pairs = generate_pairs(l1, l2, subs, list(test_pool), N_TEST_DD,
                               random.Random(SEED+2), f"{safe}_test_dd")
        ps["test_dd"] = save_jsonl(pairs, out_dir / "test_dd.jsonl")
        print(f"    test_dd:  {ps['test_dd']}")

        pairs = generate_pairs(l1, l2, subs, list(val_pool), N_VAL,
                               random.Random(SEED+3), f"{safe}_val")
        ps["val"] = save_jsonl(pairs, out_dir / "val.jsonl")
        print(f"    val:      {ps['val']}")

        if (l1, l2) in TRAIN_A_PAIRS:
            pairs = generate_pairs(l1, l2, subs, list(train_pool), N_TRAIN_A,
                                   random.Random(SEED+10), f"{safe}_train_a")
            ps["train_a"] = save_jsonl(pairs, out_dir / "train_a.jsonl")
            print(f"    train_a:  {ps['train_a']}")

        if (l1, l2) in TRAIN_B_PAIRS:
            pairs = generate_pairs(l1, l2, subs, list(train_pool), N_TRAIN_B,
                                   random.Random(SEED+20), f"{safe}_train_b")
            ps["train_b"] = save_jsonl(pairs, out_dir / "train_b.jsonl")
            print(f"    train_b:  {ps['train_b']}")

        if (l1, l2) in WEEK1_PAIRS:
            sd = generate_pairs(l1, l2, subs, list(train_pool), N_WEEK1,
                                random.Random(SEED+30), f"{safe}_zs_sd")
            dd = generate_pairs(l1, l2, subs, list(test_pool),  N_WEEK1,
                                random.Random(SEED+31), f"{safe}_zs_dd")
            zs = sd[:N_WEEK1//2] + dd[:N_WEEK1//2]
            random.Random(SEED+32).shuffle(zs)
            ps["zeroshot_w1"] = save_jsonl(zs, out_dir / "zeroshot_w1.jsonl")
            print(f"    week1_zs: {ps['zeroshot_w1']}")

        summary[safe] = ps

    (DATASET_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    total = 0
    for pk, ps in summary.items():
        counts = {k: v for k, v in ps.items() if k not in ("l1", "l2")}
        print(f"  {ps['l1']:8s} ↔ {ps['l2']:8s}: " +
              "  ".join(f"{k}={v}" for k, v in counts.items()))
        total += sum(counts.values())
    print(f"\n  TOTAL PAIRS: {total:,}")
    print(f"\nOutputs: {DATASET_DIR}/")


if __name__ == "__main__":
    main()
