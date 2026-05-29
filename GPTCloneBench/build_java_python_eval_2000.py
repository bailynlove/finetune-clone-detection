#!/usr/bin/env python3
"""
Expand GPTCloneBench Java-Python eval to 2000 pairs (1000 pos + 1000 neg).
Keeps the same seed/shuffle as the original 1000-pair build, so the first
500 pos / 500 neg pairs are identical (same code, same pair_ids).
New pairs get ids pos_000500..pos_000999 and neg_000500..neg_000999.

Outputs:
  dataset/finetune_data/{b1,b1ctrl,b2,b3}/gptclone_java_python_2000/test.jsonl
  dataset/splits/gptclonebench_java_python_2000/test.jsonl  (raw, no inline code)
"""
import json, random, zipfile
from pathlib import Path

random.seed(42)

ROOT      = Path("/data1/clone-test")
OUT_DIR   = ROOT / "dataset/finetune_data"
CODE_DIR  = ROOT / "dataset/gptclonebench_code"
SPLIT_DIR = ROOT / "dataset/splits/gptclonebench_java_python_2000"

SEP  = '$' * 40
DASH = '-' * 40

SYSTEM_MSG  = ("You are a code analysis assistant. "
               "Determine functional equivalence of code snippets across programming languages.")
INSTRUCTION = ("Determine whether the following two code snippets implement the same "
               "functionality, regardless of the programming language used. "
               'Answer with only "Yes" or "No".')


def is_python(code):
    c = code.lstrip()
    return (c.startswith('def ') or
            ('def ' in c and ':' in c and '{' not in c.split('def ')[0]))


def parse_pairs(zip_path, txt_name):
    with zipfile.ZipFile(zip_path) as z:
        txt = z.read(txt_name).decode(errors='replace')
    pairs = []
    for block in txt.split(SEP):
        block = block.strip()
        if not block:
            continue
        parts = block.split(DASH)
        if len(parts) < 3:
            continue
        c1, c2 = parts[1].strip(), parts[2].strip()
        if is_python(c2) and not is_python(c1):
            pairs.append({'java': c1, 'python': c2})
        elif is_python(c1) and not is_python(c2):
            pairs.append({'java': c2, 'python': c1})
    return pairs


print("Parsing GPTCloneBench Java-Python pairs...")
all_pos = parse_pairs(
    ROOT / "GPTCloneBench/cross_language.zip",
    "java_to_other/1_java_to_other_validateClones.txt"
)
print(f"  Found {len(all_pos)} Java-Python positive pairs")

random.shuffle(all_pos)
pos_sample = all_pos[:1000]

# 1000 negative pairs: greedy cross-pairing (same logic as original, limit doubled)
neg_sample = []
used = set()
for i in range(len(all_pos)):
    for j in range(len(all_pos)):
        if i != j and (i, j) not in used:
            neg_sample.append({'java': all_pos[i]['java'], 'python': all_pos[j]['python']})
            used.add((i, j))
            if len(neg_sample) >= 1000:
                break
    if len(neg_sample) >= 1000:
        break

print(f"  Generated {len(neg_sample)} negative pairs")


def make_record(idx, label, java_code, py_code):
    pair_id  = f"gptclone_java_python_{'pos' if label==1 else 'neg'}_{idx:06d}"
    java_rel = f"gptclonebench/java/{pair_id}_java.png"
    py_rel   = f"gptclonebench/python/{pair_id}_py.png"
    java_path = CODE_DIR / "java"   / f"{pair_id}_java.java"
    py_path   = CODE_DIR / "python" / f"{pair_id}_py.py"
    java_path.parent.mkdir(parents=True, exist_ok=True)
    py_path.parent.mkdir(parents=True, exist_ok=True)
    # Only write if the file doesn't already exist (first 500 pairs are unchanged)
    if not java_path.exists():
        java_path.write_text(java_code, encoding='utf-8')
    if not py_path.exists():
        py_path.write_text(py_code, encoding='utf-8')
    return {
        "pair_id": pair_id,
        "label":   label,
        "code1": {
            "lang": "Java", "problem_id": "gptclone",
            "submission_id": f"{pair_id}_java",
            "n_lines": len(java_code.splitlines()),
            "max_chars": max((len(l) for l in java_code.splitlines()), default=0),
            "rel_path": f"gptclonebench/java/{pair_id}_java.java",
            "image_rel_path": java_rel,
        },
        "code2": {
            "lang": "Python", "problem_id": "gptclone",
            "submission_id": f"{pair_id}_py",
            "n_lines": len(py_code.splitlines()),
            "max_chars": max((len(l) for l in py_code.splitlines()), default=0),
            "rel_path": f"gptclonebench/python/{pair_id}_py.py",
            "image_rel_path": py_rel,
        },
        "java_code":   java_code,
        "python_code": py_code,
    }


print("Writing code files (skipping existing)...")
records = []
for i, p in enumerate(pos_sample):
    records.append(make_record(i, 1, p['java'], p['python']))
for i, p in enumerate(neg_sample):
    records.append(make_record(i, 0, p['java'], p['python']))

# Shuffle with a different seed so the combined set is well-mixed
random.seed(123)
random.shuffle(records)
print(f"Total records: {len(records)}")

# Raw split (no inline code)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
raw_out = SPLIT_DIR / "test.jsonl"
with open(raw_out, 'w') as f:
    for r in records:
        row = {k: v for k, v in r.items() if k not in ('java_code', 'python_code')}
        f.write(json.dumps(row) + '\n')
print(f"Raw split: {raw_out}")

# Finetune format for all 4 modes
for mode in ['b1', 'b1ctrl', 'b2', 'b3']:
    out = OUT_DIR / mode / "gptclone_java_python_2000" / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for r in records:
            java_code = r['java_code']
            py_code   = r['python_code']
            label     = r['label']

            if mode == 'b1':
                user_content = (
                    f"{INSTRUCTION}\n\n"
                    f"Code 1 (language: Java):\n```java\n{java_code}\n```\n\n"
                    f"Code 2 (language: Python):\n```python\n{py_code}\n```\n\nAnswer:"
                )
                messages = [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": "Yes" if label == 1 else "No"},
                ]
                img = None
            else:
                if mode == 'b3':
                    user_content = [
                        {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: Java):"},
                        {"type": "image"},
                        {"type": "text",  "text": "\n\nCode 2 (language: Python):"},
                        {"type": "image"},
                        {"type": "text",  "text": "\n\nAnswer:"},
                    ]
                else:
                    user_content = [
                        {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: Java):"},
                        {"type": "image"},
                        {"type": "text",  "text": f"```java\n{java_code}\n```\n\nCode 2 (language: Python):"},
                        {"type": "image"},
                        {"type": "text",  "text": f"```python\n{py_code}\n```\n\nAnswer:"},
                    ]
                messages = [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": "Yes" if label == 1 else "No"},
                ]
                if mode == 'b1ctrl':
                    img = ["__black__", "__black__"]
                else:
                    img = [r['code1']['image_rel_path'], r['code2']['image_rel_path']]

            f.write(json.dumps({
                "pair_id":     r['pair_id'],
                "label":       label,
                "answer":      "Yes" if label == 1 else "No",
                "lang_pair":   "Java-Python",
                "messages":    messages,
                "image_paths": img,
            }) + '\n')
    n = sum(1 for _ in open(out))
    print(f"[{mode}] gptclone_java_python_2000/test.jsonl: {n} pairs")

print("\nDone. New code files written to:", CODE_DIR)
print("Run render_gptclone.py to render new images (existing images will be skipped).")
