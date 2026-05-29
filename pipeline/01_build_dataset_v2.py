#!/usr/bin/env python3
"""
01_build_dataset_v2.py — Redesigned CodeNet pair construction.

FIX for label-leakage-via-problem_id:
  Old design: positives = same-problem AC×AC, negatives = diff-problem AC×AC
              → label 100% predictable from (prob1 == prob2) — model never reads code.

  New design: 3 pair types
    POS        (label=1) AC(l1,P) × AC(l2,P)   — same problem, both correct
    HARD_NEG   (label=0) AC(l1,P) × WA(l2,P)   — same problem, one wrong answer
                      OR WA(l1,P) × AC(l2,P)
    EASY_NEG   (label=0) AC(l1,P1) × AC(l2,P2) — different problems, both correct

  Mix: 50% POS + 25% HARD_NEG + 25% EASY_NEG
  → same-problem pairs now carry ambiguous label (67% pos / 33% neg among same-prob pairs)
  → model must read code to distinguish correct from incorrect submission

Output: dataset/splits_v2/{lang_pair}/{train_a,val,test_sd,test_dd,...}.jsonl
Images for WA submissions are rendered inline at the end.
"""

import json, csv, random, os, re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import ImageFormatter
from pygments.style import Style
from pygments.token import Token

CODENET_DIR = Path("/data1/clone-test/Project_CodeNet")
DATASET_DIR = Path("/data1/clone-test/dataset")
SPLITS_DIR  = DATASET_DIR / "splits_v2"
IMAGES_DIR  = DATASET_DIR / "images"        # same image dir as v1 (shared)
SEED        = 42

MIN_LINES  = 4
MAX_LINES  = 74
MAX_CHARS  = 148
MAX_AC_SUBS = 3    # max AC submissions kept per (problem, lang)
MAX_WA_SUBS = 3    # max WA submissions kept per (problem, lang)
CANDIDATES  = 10   # number of smallest-code-size candidates to read per (prob, lang, status)

TARGET_LANGS = ["Python", "Java", "Rust", "Ruby"]

LANG_PAIRS = [
    ("Python", "Java"),
    ("Rust",   "Java"),
    ("Rust",   "Python"),
    ("Rust",   "Ruby"),
]

TRAIN_A_PAIRS = {("Python", "Java")}
TRAIN_B_PAIRS = {("Python", "Java"), ("Rust", "Java")}

N_TRAIN_A  = 20_000
N_TRAIN_B  = 10_000
N_VAL      =  3_000
N_TEST_SD  =  1_000
N_TEST_DD  =  1_000

POS_FRAC       = 0.50
HARD_NEG_FRAC  = 0.25
EASY_NEG_FRAC  = 0.25

LENGTH_BUCKETS = [(1, 15), (16, 30), (31, 50), (51, MAX_LINES)]

# ── Image rendering ───────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1440, 896
BG_COLOR   = "#272822"
IMAGE_PAD  = 10
FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

class UnifiedMonokaiStyle(Style):
    background_color = BG_COLOR
    highlight_color  = "#49483e"
    styles = {
        Token:                  "#f8f8f2",
        Token.Keyword:          "#66d9ef",
        Token.Keyword.Type:     "#66d9ef",
        Token.Name.Builtin:     "#a6e22e",
        Token.Name.Function:    "#a6e22e",
        Token.Name.Class:       "#a6e22e",
        Token.Literal.String:   "#e6db74",
        Token.Literal.Number:   "#ae81ff",
        Token.Comment:          "#75715e italic",
        Token.Operator:         "#f92672",
        Token.Punctuation:      "#f8f8f2",
    }

LANG_TO_LEXER = {
    "Python": "python", "Java": "java",
    "Rust": "rust", "Ruby": "ruby",
    "C++": "cpp",
}

def pick_font(n_lines, max_chars):
    font = FONT_TIERS[-1][1]
    for lim, pt in FONT_TIERS:
        if n_lines <= lim:
            font = pt
            break
    if max_chars > MAX_CHARS:
        pts = [pt for _, pt in FONT_TIERS]
        idx = pts.index(font)
        font = pts[min(idx + 1, len(pts) - 1)]
    return font

def render_one(args):
    src_path, out_path, lang = args
    out_path = Path(out_path)
    if out_path.exists():
        return "exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = Path(src_path).read_text(errors="replace")
    except Exception as e:
        return f"read_error:{e}"
    code = re.sub(r'[^\x00-\x7F]', ' ', raw)
    lines = code.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + ["# ... <truncated>"]
    if not lines:
        return "empty"
    code = "\n".join(lines)
    n_lines = len(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)
    font = pick_font(n_lines, max_chars)
    line_pad = max(2, int(font * 0.2))
    try:
        lexer = get_lexer_by_name(LANG_TO_LEXER.get(lang, "text"), stripall=True)
    except Exception:
        lexer = TextLexer()
    fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=font,
                         style=UnifiedMonokaiStyle, line_numbers=False,
                         image_pad=IMAGE_PAD, line_pad=line_pad)
    try:
        rendered = Image.open(BytesIO(highlight(code, lexer, fmt))).convert("RGB")
    except Exception as e:
        return f"render_error:{e}"
    if rendered.width > TARGET_W or rendered.height > TARGET_H:
        rendered.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), BG_COLOR)
    canvas.paste(rendered, (0, 0))
    canvas.save(out_path, "PNG")
    return "ok"


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_one_csv(args):
    """Returns (pid, {lang: {status: [(sid, ext, code_size), ...]}})."""
    pid, csv_path = args
    result = {lang: {"Accepted": [], "Wrong Answer": []} for lang in TARGET_LANGS}
    try:
        with open(csv_path, newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                lang   = row.get("language", "")
                status = row.get("status", "")
                if lang in TARGET_LANGS and status in ("Accepted", "Wrong Answer"):
                    try:
                        cs = int(row.get("code_size") or 0)
                    except ValueError:
                        cs = 0
                    result[lang][status].append(
                        (row["submission_id"], row["filename_ext"], cs)
                    )
    except Exception:
        pass
    return pid, result


def read_line_stats(args):
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


def read_sub_with_status(args):
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    pid, lang, sid, ext, status = args
    r = read_line_stats((pid, lang, sid, ext))
    if r is None:
        return None
    _, _, sid_r, ext_r, n, m = r
    return (pid, lang, sid_r, ext_r, n, m, status)


def filter_subs_by_exclusion(subs_dict, excluded_sids):
    """Return a new dict with any submission_id in excluded_sids removed."""
    result = defaultdict(list)
    for pid, entries in subs_dict.items():
        clean = [e for e in entries if e[0] not in excluded_sids]
        if clean:
            result[pid] = clean
    return result


def collect_sids_from_jsonl(path):
    """Collect all submission_ids (code1 + code2) from a JSONL file."""
    sids = set()
    try:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                sids.add(row["code1"]["submission_id"])
                sids.add(row["code2"]["submission_id"])
    except Exception:
        pass
    return sids


# ── Helpers ───────────────────────────────────────────────────────────────────

def bucket_of(n_lines):
    for i, (lo, hi) in enumerate(LENGTH_BUCKETS):
        if lo <= n_lines <= hi:
            return i
    return len(LENGTH_BUCKETS) - 1


def stratified_sample(items, n_target, key_fn, rng):
    n_buckets = len(LENGTH_BUCKETS)
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


def pick_subs(sub_dict, pid, n, rng):
    pool = sub_dict.get(pid, [])
    return rng.sample(pool, min(n, len(pool))) if pool else []


def mk_entry(lang, pid, sid, ext, n_lines, max_chars, is_accepted):
    return {
        "lang": lang,
        "problem_id": pid,
        "submission_id": sid,
        "is_accepted": is_accepted,
        "n_lines": n_lines,
        "max_chars": max_chars,
        "rel_path": f"data/{pid}/{lang}/{sid}.{ext}",
        "image_rel_path": f"images/{pid}/{lang}/{sid}.png",
    }


# ── Pair generation ───────────────────────────────────────────────────────────

def generate_pairs_v2(l1, l2, ac_subs, wa_subs, pool_probs, n_target, rng, tag):
    """
    50% POS + 25% HARD_NEG + 25% EASY_NEG.
    Each entry: (s1_tuple, p1, s1_accepted, s2_tuple, p2, s2_accepted, neg_type)
    """
    pool_set = set(pool_probs)

    # Problems with AC in both langs (for positives + easy_neg source)
    both_ac = [p for p in pool_probs if ac_subs[l1].get(p) and ac_subs[l2].get(p)]
    # Problems with AC in one lang and WA in the other (for hard_neg)
    hard_src = [p for p in pool_probs
                if (ac_subs[l1].get(p) and wa_subs[l2].get(p)) or
                   (wa_subs[l1].get(p) and ac_subs[l2].get(p))]

    n_pos      = int(n_target * POS_FRAC)
    n_hard     = int(n_target * HARD_NEG_FRAC)
    n_easy     = n_target - n_pos - n_hard

    # ── Positives ─────────────────────────────────────────────────────────────
    all_pos = []
    for p in both_ac:
        s1s = pick_subs(ac_subs[l1], p, MAX_AC_SUBS, rng)
        s2s = pick_subs(ac_subs[l2], p, MAX_AC_SUBS, rng)
        for s1 in s1s:
            for s2 in s2s:
                all_pos.append((s1, p, True, s2, p, True, None, max(s1[2], s2[2])))

    sampled_pos = stratified_sample(all_pos, n_pos,
                                    key_fn=lambda x: bucket_of(x[7]), rng=rng)

    # ── Hard negatives (same problem, one WA) ─────────────────────────────────
    all_hard = []
    for p in hard_src:
        # AC(l1) × WA(l2)
        if ac_subs[l1].get(p) and wa_subs[l2].get(p):
            s1s = pick_subs(ac_subs[l1], p, MAX_AC_SUBS, rng)
            s2s = pick_subs(wa_subs[l2], p, MAX_WA_SUBS, rng)
            for s1 in s1s:
                for s2 in s2s:
                    all_hard.append((s1, p, True, s2, p, False, "hard", max(s1[2], s2[2])))
        # WA(l1) × AC(l2)
        if wa_subs[l1].get(p) and ac_subs[l2].get(p):
            s1s = pick_subs(wa_subs[l1], p, MAX_WA_SUBS, rng)
            s2s = pick_subs(ac_subs[l2], p, MAX_AC_SUBS, rng)
            for s1 in s1s:
                for s2 in s2s:
                    all_hard.append((s1, p, False, s2, p, True, "hard", max(s1[2], s2[2])))

    sampled_hard = stratified_sample(all_hard, n_hard,
                                     key_fn=lambda x: bucket_of(x[7]), rng=rng)

    # ── Easy negatives (different problem, both AC) ────────────────────────────
    all_l1 = [(s, p) for p in both_ac for s in ac_subs[l1].get(p, [])]
    all_l2 = [(s, p) for p in both_ac for s in ac_subs[l2].get(p, [])]
    seen, all_easy = set(), []
    for _ in range(n_easy * 40):
        if len(all_easy) >= n_easy * 3:
            break
        s1, p1 = rng.choice(all_l1)
        s2, p2 = rng.choice(all_l2)
        if p1 != p2:
            key = (s1[0], s2[0])
            if key not in seen:
                seen.add(key)
                all_easy.append((s1, p1, True, s2, p2, True, "easy", max(s1[2], s2[2])))

    sampled_easy = stratified_sample(all_easy, n_easy,
                                     key_fn=lambda x: bucket_of(x[7]), rng=rng)

    # ── Assemble records ──────────────────────────────────────────────────────
    pos_count = neg_hard_count = neg_easy_count = 0
    records = []
    for idx, group in enumerate([
        (1,  sampled_pos,  "pos"),
        (0,  sampled_hard, "hard"),
        (0,  sampled_easy, "easy"),
    ]):
        label, items, kind = group
        for i, (s1, p1, ac1, s2, p2, ac2, neg_type, _) in enumerate(items):
            sid1, ext1, nl1, mc1 = s1
            sid2, ext2, nl2, mc2 = s2
            rec = {
                "pair_id":  f"{tag}_{kind}_{i:06d}",
                "label":    label,
                "neg_type": neg_type,
                "code1":    mk_entry(l1, p1, sid1, ext1, nl1, mc1, ac1),
                "code2":    mk_entry(l2, p2, sid2, ext2, nl2, mc2, ac2),
            }
            records.append(rec)
        if kind == "pos":
            pos_count = len(items)
        elif kind == "hard":
            neg_hard_count = len(items)
        else:
            neg_easy_count = len(items)

    rng.shuffle(records)
    print(f"      pos={pos_count}  hard_neg={neg_hard_count}  easy_neg={neg_easy_count}  "
          f"total={len(records)}  (requested {n_target})")
    if len(records) < n_target * 0.8:
        print(f"      WARNING: only generated {len(records)}/{n_target} pairs")
    return records


def save_jsonl(pairs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return len(pairs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Parse CSVs
    print("Step 1: Parsing metadata CSVs (AC + WA)...")
    meta_dir = CODENET_DIR / "metadata"
    all_csv  = sorted(meta_dir.glob("*.csv"))

    raw_ac = {lang: defaultdict(list) for lang in TARGET_LANGS}
    raw_wa = {lang: defaultdict(list) for lang in TARGET_LANGS}

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(parse_one_csv, (p.stem, str(p))): p.stem for p in all_csv}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing CSVs"):
            pid, result = fut.result()
            for lang in TARGET_LANGS:
                raw_ac[lang][pid] = result[lang]["Accepted"]
                raw_wa[lang][pid] = result[lang]["Wrong Answer"]

    for lang in TARGET_LANGS:
        ac_total = sum(len(v) for v in raw_ac[lang].values())
        wa_total = sum(len(v) for v in raw_wa[lang].values())
        print(f"  {lang}: AC={ac_total:,}  WA={wa_total:,}")

    # Step 2: Filter by line/char limits
    print(f"\nStep 2: Filtering submissions (line/char limits)...")
    tasks = []
    for lang in TARGET_LANGS:
        all_pids = set(raw_ac[lang].keys()) | set(raw_wa[lang].keys())
        for pid in all_pids:
            for status, src in [("ac", raw_ac), ("wa", raw_wa)]:
                sub_list = src[lang].get(pid, [])
                if not sub_list:
                    continue
                candidates = sorted(sub_list, key=lambda x: x[2])[:CANDIDATES]
                for sid, ext, _cs in candidates:
                    tasks.append((pid, lang, sid, ext, status))

    print(f"  Reading {len(tasks):,} files (pool.map chunksize=200)...")

    ac_subs = {lang: defaultdict(list) for lang in TARGET_LANGS}
    wa_subs = {lang: defaultdict(list) for lang in TARGET_LANGS}

    with ProcessPoolExecutor(max_workers=8) as pool:
        for res in tqdm(pool.map(read_sub_with_status, tasks, chunksize=200),
                        total=len(tasks), desc="Reading"):
            if res is None:
                continue
            pid, lang, sid, ext, n, m, status = res
            if MIN_LINES <= n <= MAX_LINES and m <= MAX_CHARS:
                entry = (sid, ext, n, m)
                if status == "ac":
                    ac_subs[lang][pid].append(entry)
                else:
                    wa_subs[lang][pid].append(entry)

    for lang in TARGET_LANGS:
        ac_p = len(ac_subs[lang])
        wa_p = len(wa_subs[lang])
        ac_s = sum(len(v) for v in ac_subs[lang].values())
        wa_s = sum(len(v) for v in wa_subs[lang].values())
        print(f"  {lang}: AC {ac_p} probs/{ac_s} subs  |  WA {wa_p} probs/{wa_s} subs")

    # Step 3: Problem pool split
    print("\nStep 3: Problem pool split...")
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

    print(f"  train={len(train_pool):,}  val={len(val_pool):,}  test(DD)={len(test_pool):,}")

    problem_splits = {
        "train_pool": sorted(train_pool),
        "val_pool":   sorted(val_pool),
        "test_pool":  sorted(test_pool),
    }
    (DATASET_DIR / "problem_splits_v2.json").write_text(json.dumps(problem_splits, indent=2))

    # Step 4: Generate pairs
    print("\nStep 4: Generating pairs (50% pos / 25% hard_neg / 25% easy_neg)...")
    summary = {}
    wa_images_needed = []   # (src_path, out_path, lang) for WA submissions

    for l1, l2 in LANG_PAIRS:
        safe = f"{l1.lower().replace('+','p')}_{l2.lower()}"
        out_dir = SPLITS_DIR / safe
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{l1} ↔ {l2}]")

        def overlap_stats(pool):
            ac_both = sum(1 for p in pool if ac_subs[l1].get(p) and ac_subs[l2].get(p))
            hard_ok = sum(1 for p in pool
                          if (ac_subs[l1].get(p) and wa_subs[l2].get(p)) or
                             (wa_subs[l1].get(p) and ac_subs[l2].get(p)))
            return ac_both, hard_ok

        ac_b, h = overlap_stats(train_pool)
        print(f"    train_pool: {ac_b} probs with AC×AC, {h} probs with AC×WA")
        ac_b, h = overlap_stats(test_pool)
        print(f"    test_pool:  {ac_b} probs with AC×AC, {h} probs with AC×WA")

        ps = {"l1": l1, "l2": l2}

        # ── Generate training data FIRST so we can exclude those submissions ──
        train_excluded = set()   # submission_ids seen in training (for this lang pair)

        if (l1, l2) in TRAIN_A_PAIRS:
            print(f"    train_a:")
            p = generate_pairs_v2(l1, l2, ac_subs, wa_subs, list(train_pool),
                                   N_TRAIN_A, random.Random(SEED + 10), f"{safe}_train_a")
            ps["train_a"] = save_jsonl(p, out_dir / "train_a.jsonl")
            train_excluded |= collect_sids_from_jsonl(out_dir / "train_a.jsonl")
            print(f"      → excluding {len(train_excluded):,} submission_ids from test_sd")

        if (l1, l2) in TRAIN_B_PAIRS:
            print(f"    train_b:")
            p = generate_pairs_v2(l1, l2, ac_subs, wa_subs, list(train_pool),
                                   N_TRAIN_B, random.Random(SEED + 20), f"{safe}_train_b")
            ps["train_b"] = save_jsonl(p, out_dir / "train_b.jsonl")

        # ── Build per-pair ac/wa dicts with training submissions excluded ──────
        ac_test = {lang: filter_subs_by_exclusion(ac_subs[lang], train_excluded)
                   for lang in TARGET_LANGS}
        wa_test = {lang: filter_subs_by_exclusion(wa_subs[lang], train_excluded)
                   for lang in TARGET_LANGS}

        # ── test_sd (train_pool problems, filtered submissions) ────────────────
        print(f"    test_sd (submission-disjoint from train_a):")
        p = generate_pairs_v2(l1, l2, ac_test, wa_test, list(train_pool),
                               N_TEST_SD, random.Random(SEED + 1), f"{safe}_test_sd")
        ps["test_sd"] = save_jsonl(p, out_dir / "test_sd.jsonl")

        # ── test_dd (test_pool problems — completely unseen problems) ──────────
        print(f"    test_dd:")
        p = generate_pairs_v2(l1, l2, ac_subs, wa_subs, list(test_pool),
                               N_TEST_DD, random.Random(SEED + 2), f"{safe}_test_dd")
        ps["test_dd"] = save_jsonl(p, out_dir / "test_dd.jsonl")

        # ── val (val_pool problems — unseen problems, no filtering needed) ──────
        print(f"    val:")
        p = generate_pairs_v2(l1, l2, ac_subs, wa_subs, list(val_pool),
                               N_VAL, random.Random(SEED + 3), f"{safe}_val")
        ps["val"] = save_jsonl(p, out_dir / "val.jsonl")

        summary[safe] = ps

        # Collect WA submission image tasks for this lang pair
        for split_name in ps:
            if split_name in ("l1", "l2"):
                continue
            jsonl = out_dir / f"{split_name}.jsonl"
            with open(jsonl) as f:
                for line in f:
                    row = json.loads(line)
                    for code_key in ("code1", "code2"):
                        c = row[code_key]
                        if not c["is_accepted"]:
                            src = CODENET_DIR / c["rel_path"]
                            out = DATASET_DIR / c["image_rel_path"]
                            wa_images_needed.append(
                                (str(src), str(out), c["lang"])
                            )

    # Deduplicate image tasks
    wa_images_needed = list({(s, o): (s, o, l)
                             for s, o, l in wa_images_needed}.values())

    # Step 5: Render WA submission images
    print(f"\nStep 5: Rendering {len(wa_images_needed):,} WA submission images "
          f"(skipping existing)...")
    ok = err = skip = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(render_one, t): t for t in wa_images_needed}
        for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs))):
            status = fut.result()
            if status == "ok":      ok   += 1
            elif status == "exists": skip += 1
            else:                   err  += 1
    print(f"  Rendered: ok={ok}  err={err}  skip={skip}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET V2 SUMMARY")
    print("=" * 60)
    for pk, ps in summary.items():
        counts = {k: v for k, v in ps.items() if k not in ("l1", "l2")}
        print(f"  {ps['l1']:8s} ↔ {ps['l2']:8s}: " +
              "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"\nOutputs: {SPLITS_DIR}/")
    print(f"WA images: {ok} new renders, {skip} already existed, {err} errors")

    (DATASET_DIR / "dataset_summary_v2.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
