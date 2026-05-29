#!/usr/bin/env python3
"""
00_prepare_data_v2.py
Convert splits_v2 JSONL pairs → fine-tuning examples for B1 / B1-ctrl / B2 / B3.

New in v2:
  - Source: dataset/splits_v2/{lang_pair}/ (includes hard negatives)
  - Output records include neg_type: null | "hard" | "easy"
  - is_accepted flag carried in metadata (not shown to model)

Same pair types as v1:
  B1      text-only
  B1-ctrl black image + text (confound control)
  B2      real image + text  (main method)
  B3      real image-only

Output: dataset/finetune_data_v2/{b1,b1ctrl,b2,b3}/{split}/{lang_pair}.jsonl
"""

import json
from pathlib import Path

DATASET_DIR = Path("/data1/clone-test/dataset")
SPLITS_DIR  = DATASET_DIR / "splits_v2"
OUT_DIR     = DATASET_DIR / "finetune_data_v2"
CODENET_DIR = Path("/data1/clone-test/Project_CodeNet")

SYSTEM_MSG = (
    "You are a code analysis assistant. "
    "Determine functional equivalence of code snippets across programming languages."
)

INSTRUCTION = (
    "Determine whether the following two code snippets implement the same "
    "functionality, regardless of the programming language used. "
    'Answer with only "Yes" or "No".'
)


def read_code(code_entry):
    rel = code_entry["rel_path"]
    path = CODENET_DIR / rel
    try:
        return path.read_text(errors="replace").strip()
    except Exception:
        return ""


def make_b1_messages(p):
    c1, c2 = p["code1"], p["code2"]
    text1, text2 = read_code(c1), read_code(c2)
    user_content = (
        f"{INSTRUCTION}\n\n"
        f"Code 1 (language: {c1['lang']}):\n"
        f"```{c1['lang'].lower()}\n{text1}\n```\n\n"
        f"Code 2 (language: {c2['lang']}):\n"
        f"```{c2['lang'].lower()}\n{text2}\n```\n\n"
        "Answer:"
    )
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": "Yes" if p["label"] == 1 else "No"},
    ]


def make_b1ctrl_messages(p):
    c1, c2 = p["code1"], p["code2"]
    text1, text2 = read_code(c1), read_code(c2)
    user_content = [
        {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: {c1['lang']}):"},
        {"type": "image"},
        {"type": "text",  "text": f"```{c1['lang'].lower()}\n{text1}\n```\n\nCode 2 (language: {c2['lang']}):"},
        {"type": "image"},
        {"type": "text",  "text": f"```{c2['lang'].lower()}\n{text2}\n```\n\nAnswer:"},
    ]
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": "Yes" if p["label"] == 1 else "No"},
    ]


def make_b2_messages(p):
    return make_b1ctrl_messages(p)


def make_b3_messages(p):
    c1, c2 = p["code1"], p["code2"]
    user_content = [
        {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: {c1['lang']}):"},
        {"type": "image"},
        {"type": "text",  "text": f"\n\nCode 2 (language: {c2['lang']}):"},
        {"type": "image"},
        {"type": "text",  "text": "\n\nAnswer:"},
    ]
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": "Yes" if p["label"] == 1 else "No"},
    ]


MAKERS = {
    "b1":     (make_b1_messages,     "none"),
    "b1ctrl": (make_b1ctrl_messages, "black"),
    "b2":     (make_b2_messages,     "paths"),
    "b3":     (make_b3_messages,     "paths"),
}

SPLIT_MAP = {
    "train_a.jsonl": "train_a",
    "train_b.jsonl": "train_b",
    "val.jsonl":     "val",
    "test_sd.jsonl": "test_sd",
    "test_dd.jsonl": "test_dd",
}


def convert_file(jsonl_path, make_fn, image_mode, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(jsonl_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            try:
                p = json.loads(line)
            except Exception:
                continue
            messages = make_fn(p)
            if image_mode == "none":
                image_paths = None
            elif image_mode == "black":
                image_paths = ["__black__", "__black__"]
            else:
                image_paths = [p["code1"]["image_rel_path"], p["code2"]["image_rel_path"]]

            record = {
                "pair_id":     p["pair_id"],
                "label":       p["label"],
                "neg_type":    p.get("neg_type"),   # null | "hard" | "easy"
                "answer":      "Yes" if p["label"] == 1 else "No",
                "lang_pair":   f"{p['code1']['lang']}-{p['code2']['lang']}",
                "messages":    messages,
                "image_paths": image_paths,
            }
            fout.write(json.dumps(record) + "\n")
            written += 1
    return written


def main():
    all_jsonl = sorted(SPLITS_DIR.rglob("*.jsonl"))
    print(f"Found {len(all_jsonl)} JSONL files under {SPLITS_DIR}")

    for mode, (make_fn, image_mode) in MAKERS.items():
        print(f"\n── mode={mode}  image_mode={image_mode} ──")
        totals = {}
        for src in all_jsonl:
            split_name = SPLIT_MAP.get(src.name)
            if split_name is None:
                continue
            lang_pair = src.parent.name
            out_path  = OUT_DIR / mode / split_name / f"{lang_pair}.jsonl"
            n = convert_file(src, make_fn, image_mode, out_path)
            totals[split_name] = totals.get(split_name, 0) + n

        for split, n in sorted(totals.items()):
            print(f"  {split}: {n:,} examples")

    print("\nDone. Output:", OUT_DIR)


if __name__ == "__main__":
    main()
