#!/usr/bin/env python3
"""
00_prepare_data.py
Convert dataset JSONL pairs → fine-tuning examples for B1 / B1-ctrl / B2 / B3.

Per finetune_plan_v2.md §3.3 — same-model-different-input design:
  B1      text-only          (VLM receives no image token)
  B1-ctrl black image + text (image token present, but content is black — confound control)
  B2      real image + text  (main method)
  B3      real image-only    (no code text in prompt)

Output: dataset/finetune_data/{b1,b1ctrl,b2,b3}/{split}/{lang_pair}.jsonl
Each record:
  {
    "pair_id":    "...",
    "label":      0|1,
    "answer":     "Yes"|"No",
    "lang_pair":  "python-java",
    "messages":   [...],        # OpenAI-style chat with system turn
    "image_paths": null | ["__black__","__black__"] | ["rel/path1","rel/path2"]
  }

Images are NOT embedded — training script loads them by path.
image_paths="__black__" means the collator generates a black 1440×896 image at runtime.
"""

import json
from pathlib import Path

DATASET_DIR = Path("/data1/clone-test/dataset")
SPLITS_DIR  = DATASET_DIR / "splits"
OUT_DIR     = DATASET_DIR / "finetune_data"
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


def read_code(pair_entry):
    rel = pair_entry["rel_path"]
    path = CODENET_DIR / rel
    try:
        return path.read_text(errors="replace").strip()
    except Exception:
        return ""


# ── Message builders ──────────────────────────────────────────────────────────

def make_b1_messages(p):
    """B1 — text-only: VLM receives no image token."""
    c1, c2 = p["code1"], p["code2"]
    text1 = read_code(c1)
    text2 = read_code(c2)
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
    """B1-ctrl — black image + text: same prompt as B2 but with synthetic black images."""
    c1, c2 = p["code1"], p["code2"]
    text1 = read_code(c1)
    text2 = read_code(c2)
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
    """B2 — real image + text (main method). Same prompt structure as B1-ctrl."""
    return make_b1ctrl_messages(p)


def make_b3_messages(p):
    """B3 — image-only: no code text in prompt."""
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


# ── Mode → (message_fn, image_mode) ──────────────────────────────────────────
# image_mode:  "none"  → image_paths = null
#              "black" → image_paths = ["__black__", "__black__"]
#              "paths" → image_paths = [code1.image_rel_path, code2.image_rel_path]

MAKERS = {
    "b1":     (make_b1_messages,     "none"),
    "b1ctrl": (make_b1ctrl_messages, "black"),
    "b2":     (make_b2_messages,     "paths"),
    "b3":     (make_b3_messages,     "paths"),
}

SPLIT_MAP = {
    "train_a.jsonl":      "train_a",
    "train_b.jsonl":      "train_b",
    "val.jsonl":          "val",
    "test_sd.jsonl":      "test_sd",
    "test_dd.jsonl":      "test_dd",
    "zeroshot_w1.jsonl":  "zeroshot_w1",
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
