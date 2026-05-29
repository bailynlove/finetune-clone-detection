#!/usr/bin/env python3
"""
Resample deduped test sets back to 1000 pairs each (500 pos / 500 neg).

- test_sd:  draw from train_b.jsonl (clean, same-distribution candidates)
- test_dd:  generate new pairs from test_pool problems (unseen problems)

All new candidates pass the same dedup criteria:
  Stage 1: submission ID not in train_a
  Stage 2: code Jaccard similarity to train_a codes < 0.7
"""
import ast, json, re, random, itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)

CODENET  = Path("/data1/clone-test/Project_CodeNet/data")
SPLITS   = Path("/data1/clone-test/dataset/splits/python_java")
FINETUNE = Path("/data1/clone-test/dataset/finetune_data")

# ── Hashing / similarity ───────────────────────────────────────────────────────

def load_code(rel):
    try: return (Path("/data1/clone-test/Project_CodeNet") / rel).read_text(errors="replace")
    except: return ""

def normalize(code):
    code = re.sub(r'#[^\n]*', '', code)
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', code).strip().lower()

def shingles(text, k=5):
    return frozenset(text[i:i+k] for i in range(max(0, len(text)-k+1)))

def jaccard(a, b):
    u = len(a | b); return len(a & b)/u if u else (1.0 if not a and not b else 0.0)

N_PERM=128; rng=np.random.RandomState(42)
A=rng.randint(1,2**31-1,N_PERM,dtype=np.int64)
B=rng.randint(0,2**31-1,N_PERM,dtype=np.int64)
P=np.int64(2**31-1)

def minhash(shin):
    sig=np.full(N_PERM,P,dtype=np.int64)
    for s in shin:
        h=hash(s)%P; sig=np.minimum(sig,(A*h+B)%P)
    return sig

def build_lsh(sigs,nb=16):
    rows=N_PERM//nb; bkts=defaultdict(list)
    for i,sig in enumerate(sigs):
        for b in range(nb): bkts[(b,tuple(sig[b*rows:(b+1)*rows]))].append(i)
    return bkts,rows

def query_lsh(sig,bkts,rows,nb=16):
    cands=set()
    for b in range(nb): cands.update(bkts.get((b,tuple(sig[b*rows:(b+1)*rows])),[]))
    return cands

def is_clean(c1_meta, c2_meta, train_subs, train_shin_list, bkts, rows, thresh=0.7):
    for c in [c1_meta, c2_meta]:
        if c['submission_id'] in train_subs:
            return False
    for c in [c1_meta, c2_meta]:
        norm = normalize(load_code(c['rel_path']))
        shin = shingles(norm)
        sig  = minhash(shin)
        for idx in query_lsh(sig, bkts, rows):
            if jaccard(shin, train_shin_list[idx]) > thresh:
                return False
    return True

# ── Build dedup index from train_a ─────────────────────────────────────────────

print("Indexing train_a for dedup...")
train_subs = set()
train_shin_list = []
train_sigs_list = []
with open(SPLITS / "train_a.jsonl") as f:
    for line in f:
        r = json.loads(line)
        for key in ['code1','code2']:
            c = ast.literal_eval(r[key]) if isinstance(r[key],str) else r[key]
            train_subs.add(c['submission_id'])
            shin = shingles(normalize(load_code(c['rel_path'])))
            train_shin_list.append(shin)
            train_sigs_list.append(minhash(shin))

bkts, rows = build_lsh(train_sigs_list)
print(f"  {len(train_subs)} submissions, {len(train_sigs_list)} codes indexed")

# ── Helpers for formatting ─────────────────────────────────────────────────────

def make_code_meta(problem_id, lang, sub_path):
    suffix = 'py' if lang == 'Python' else 'java'
    sub_id = sub_path.stem
    rel = f"data/{problem_id}/{lang}/{sub_path.name}"
    img_rel = f"images/{problem_id}/{lang}/{sub_path.stem}.png"
    text = sub_path.read_text(errors='replace')
    lines = text.splitlines()
    return {
        'lang': lang,
        'problem_id': problem_id,
        'submission_id': sub_id,
        'n_lines': len(lines),
        'max_chars': max((len(l) for l in lines), default=0),
        'rel_path': rel,
        'image_rel_path': img_rel,
    }

def load_pair_ids_from_file(path):
    ids = set()
    with open(path) as f:
        for line in f:
            try: ids.add(json.loads(line)['pair_id'])
            except: pass
    return ids

# ── PART 1: Resample test_sd from train_b ─────────────────────────────────────

print("\n=== test_sd resampling from train_b ===")
with open(SPLITS / "test_sd_dedup.jsonl") as f:
    existing_sd = [json.loads(l) for l in f]
existing_sd_ids = {r['pair_id'] for r in existing_sd}

existing_pos = sum(1 for r in existing_sd if r['label'] == 1)
existing_neg = sum(1 for r in existing_sd if r['label'] == 0)
need_pos = 500 - existing_pos
need_neg = 500 - existing_neg
print(f"  Have: {existing_pos} pos, {existing_neg} neg")
print(f"  Need: {need_pos} pos, {need_neg} neg from train_b")

candidates_pos = []
candidates_neg = []
with open(SPLITS / "train_b.jsonl") as f:
    for i, line in enumerate(f):
        if i % 2000 == 0: print(f"  Scanning train_b {i}/10000...", end='\r')
        r = json.loads(line)
        if r['pair_id'] in existing_sd_ids:
            continue
        c1 = ast.literal_eval(r['code1']) if isinstance(r['code1'],str) else r['code1']
        c2 = ast.literal_eval(r['code2']) if isinstance(r['code2'],str) else r['code2']
        if is_clean(c1, c2, train_subs, train_shin_list, bkts, rows):
            if r['label'] == 1:
                candidates_pos.append(r)
            else:
                candidates_neg.append(r)

print(f"\n  Clean candidates: {len(candidates_pos)} pos, {len(candidates_neg)} neg")
random.shuffle(candidates_pos)
random.shuffle(candidates_neg)
new_sd = candidates_pos[:need_pos] + candidates_neg[:need_neg]
print(f"  Sampled: {min(need_pos,len(candidates_pos))} pos + {min(need_neg,len(candidates_neg))} neg")

final_sd = existing_sd + new_sd
final_sd.sort(key=lambda r: r['pair_id'])
print(f"  Final test_sd: {len(final_sd)} pairs ({sum(1 for r in final_sd if r['label']==1)} pos, {sum(1 for r in final_sd if r['label']==0)} neg)")

# Write
out_raw = SPLITS / "test_sd_1000.jsonl"
with open(out_raw, 'w') as f:
    for r in final_sd:
        f.write(json.dumps(r) + '\n')
print(f"  Written: {out_raw}")

# ── PART 2: Resample test_dd from test_pool ───────────────────────────────────

print("\n=== test_dd resampling from test_pool ===")
with open(SPLITS / "test_dd_dedup.jsonl") as f:
    existing_dd = [json.loads(l) for l in f]
existing_dd_pos = sum(1 for r in existing_dd if r['label'] == 1)
existing_dd_neg = sum(1 for r in existing_dd if r['label'] == 0)
need_pos_dd = 500 - existing_dd_pos
need_neg_dd = 500 - existing_dd_neg
print(f"  Have: {existing_dd_pos} pos, {existing_dd_neg} neg")
print(f"  Need: {need_pos_dd} pos, {need_neg_dd} neg from test_pool")

# Find test_pool problems and their submissions
splits_data = json.loads(open('/data1/clone-test/dataset/problem_splits.json').read())
test_pool = set(splits_data['test_pool'])

# Collect all Python and Java subs for test_pool problems
existing_dd_subs = set()
existing_dd_prob_pairs = set()
for r in existing_dd:
    c1 = ast.literal_eval(r['code1']) if isinstance(r['code1'],str) else r['code1']
    c2 = ast.literal_eval(r['code2']) if isinstance(r['code2'],str) else r['code2']
    existing_dd_subs.add(c1['submission_id'])
    existing_dd_subs.add(c2['submission_id'])
    existing_dd_prob_pairs.add((c1['problem_id'], c2['problem_id']))

print(f"  Existing dd uses {len(existing_dd_subs)} unique submissions")

# Build index of available test_pool submissions
py_subs = defaultdict(list)  # problem_id -> [sub_path]
ja_subs = defaultdict(list)
for prob_id in test_pool:
    pp = CODENET / prob_id
    if (pp / 'Python').exists():
        py_subs[prob_id] = sorted((pp / 'Python').glob('*.py'))
    if (pp / 'Java').exists():
        ja_subs[prob_id] = sorted((pp / 'Java').glob('*.java'))

# Positive pairs: same problem, one Python + one Java not already used
print("  Generating positive candidates from test_pool...")
pos_cands_dd = []
probs_with_both = [p for p in test_pool if py_subs[p] and ja_subs[p]]
random.shuffle(probs_with_both)

for prob_id in probs_with_both:
    py_list = [s for s in py_subs[prob_id] if s.stem not in existing_dd_subs]
    ja_list = [s for s in ja_subs[prob_id] if s.stem not in existing_dd_subs]
    for py_sub, ja_sub in itertools.product(py_list, ja_list):
        c1_meta = make_code_meta(prob_id, 'Python', py_sub)
        c2_meta = make_code_meta(prob_id, 'Java', ja_sub)
        if is_clean(c1_meta, c2_meta, train_subs, train_shin_list, bkts, rows):
            pos_cands_dd.append((c1_meta, c2_meta, 1))
        if len(pos_cands_dd) >= need_pos_dd * 3:
            break
    if len(pos_cands_dd) >= need_pos_dd * 3:
        break

print(f"  Positive candidates: {len(pos_cands_dd)}")

# Negative pairs: different test_pool problems
print("  Generating negative candidates from test_pool...")
neg_cands_dd = []
prob_list = [p for p in test_pool if py_subs[p] or ja_subs[p]]
random.shuffle(prob_list)

for i, p1 in enumerate(prob_list):
    for p2 in prob_list[i+1:]:
        if (p1,p2) in existing_dd_prob_pairs or (p2,p1) in existing_dd_prob_pairs:
            continue
        py_list = [s for s in py_subs.get(p1,[]) if s.stem not in existing_dd_subs]
        ja_list = [s for s in ja_subs.get(p2,[]) if s.stem not in existing_dd_subs]
        if not py_list or not ja_list:
            continue
        py_sub = random.choice(py_list)
        ja_sub = random.choice(ja_list)
        c1_meta = make_code_meta(p1, 'Python', py_sub)
        c2_meta = make_code_meta(p2, 'Java', ja_sub)
        if is_clean(c1_meta, c2_meta, train_subs, train_shin_list, bkts, rows):
            neg_cands_dd.append((c1_meta, c2_meta, 0))
        if len(neg_cands_dd) >= need_neg_dd * 3:
            break
    if len(neg_cands_dd) >= need_neg_dd * 3:
        break

print(f"  Negative candidates: {len(neg_cands_dd)}")

random.shuffle(pos_cands_dd)
random.shuffle(neg_cands_dd)
sampled_pos_dd = pos_cands_dd[:need_pos_dd]
sampled_neg_dd = neg_cands_dd[:need_neg_dd]

# Convert new dd records to jsonl format
def make_record(c1_meta, c2_meta, label, split_name, idx):
    lang = 'python_java'
    tag = 'pos' if label == 1 else 'neg'
    pair_id = f"{lang}_{split_name}_resample_{tag}_{idx:06d}"
    return {
        'pair_id': pair_id,
        'label':   label,
        'code1':   str(c1_meta),
        'code2':   str(c2_meta),
    }

new_dd_pos = [make_record(c1,c2,l,'test_dd',i) for i,(c1,c2,l) in enumerate(sampled_pos_dd)]
new_dd_neg = [make_record(c1,c2,l,'test_dd',i+1000) for i,(c1,c2,l) in enumerate(sampled_neg_dd)]
final_dd = existing_dd + new_dd_pos + new_dd_neg
final_dd.sort(key=lambda r: r['pair_id'])
print(f"  Final test_dd: {len(final_dd)} pairs ({sum(1 for r in final_dd if r['label']==1)} pos, {sum(1 for r in final_dd if r['label']==0)} neg)")

out_dd = SPLITS / "test_dd_1000.jsonl"
with open(out_dd, 'w') as f:
    for r in final_dd:
        f.write(json.dumps(r) + '\n')
print(f"  Written: {out_dd}")
print("\nDone. Use test_sd_1000.jsonl and test_dd_1000.jsonl as the clean test sets.")
