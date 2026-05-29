#!/usr/bin/env python3
"""
build_poj104_eval.py
Build evaluation pairs from CodeXGLUE POJ-104 test set, render C images,
and write finetune-format JSONL for B1 / B2 / zero-shot evaluation.

POJ-104 test: 24 problems × 500 C/C++ programs = 12,000 programs.
Pairs sampled: 500 positive (same problem) + 500 negative (different problem).

Output:
  dataset/poj104/images/<label>/<index>.png
  dataset/poj104/test_b1.jsonl
  dataset/poj104/test_b2.jsonl   (same records, image_paths filled)
"""

import json
import random
import re
import sys
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from PIL import Image
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import ImageFormatter
from pygments.style import Style
from pygments.token import (Token, Punctuation, Comment, Error, Generic,
                             Keyword, Name, Number, Operator, String)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/data1/clone-test")
POJ_DIR    = BASE_DIR / "CodeXGLUE_POJ104/Code-Code/Clone-detection-POJ-104/dataset"
OUT_DIR    = BASE_DIR / "dataset/poj104"
IMAGES_DIR = OUT_DIR / "images"

SEED        = 42
N_POS       = 500
N_NEG       = 500
TARGET_W    = 1440
TARGET_H    = 896
BG_COLOR    = (39, 40, 34)
IMAGE_PAD   = 10
MAX_LINES   = 45
MAX_CHARS   = 120

FONT_TIERS  = [(15, 26), (30, 20), (45, 16), (99, 13)]

# ── Monokai style (same as pipeline) ───────────────────────────────────────
class UnifiedMonokaiStyle(Style):
    background_color = "#272822"
    highlight_color  = "#49483e"
    default_style    = ""
    styles = {
        Token:           "#f8f8f2",
        Punctuation:     "#f8f8f2",
        Comment:         "#75715e",
        Error:           "#960050 bg:#1e0010",
        Generic.Error:   "#960050 bg:#1e0010",
        Keyword:         "#f92672",
        Number:          "#ae81ff",
        Operator:        "#f92672",
        Operator.Word:   "#f92672",
        String:          "#e6db74",
        String.Escape:   "#ae81ff",
        Name:            "#f8f8f2",
        Name.Attribute:  "#a6e22e",
        Name.Class:      "#a6e22e",
        Name.Decorator:  "#a6e22e",
        Name.Function:   "#a6e22e",
        Name.Tag:        "#f92672",
        Generic:         "#f8f8f2",
        Generic.Output:  "#66d9ef",
    }


def pick_font(n_lines, max_chars):
    font = FONT_TIERS[-1][1]
    for max_n, pt in FONT_TIERS:
        if n_lines <= max_n:
            font = pt
            break
    if max_chars > MAX_CHARS:
        pts = [pt for _, pt in FONT_TIERS]
        idx = pts.index(font)
        font = pts[min(idx + 1, len(pts) - 1)]
    return font


def prepare_code(raw):
    code = re.sub(r'[^\x00-\x7F]', ' ', raw)
    lines = code.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + ["// ... <truncated>"]
    if not lines:
        return "", 0, 0
    code = "\n".join(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)
    return code, len(lines), max_chars


def render_code_to_png(code_text, out_path: Path):
    """Render C/C++ code string → 1440×896 PNG at out_path."""
    if out_path.exists():
        return "skip"
    code, n_lines, max_chars = prepare_code(code_text)
    if n_lines == 0:
        return "empty"
    font = pick_font(n_lines, max_chars)
    line_pad = max(2, int(font * 0.2))
    try:
        lexer = get_lexer_by_name("cpp", stripall=True)
    except Exception:
        from pygments.lexers import TextLexer
        lexer = TextLexer()
    fmt = ImageFormatter(
        font_name="DejaVu Sans Mono",
        font_size=font,
        style=UnifiedMonokaiStyle,
        line_numbers=False,
        image_pad=IMAGE_PAD,
        line_pad=line_pad,
    )
    try:
        result_bytes = highlight(code, lexer, fmt)
        rendered = Image.open(BytesIO(result_bytes)).convert("RGB")
    except Exception as e:
        return f"render_error:{e}"
    if rendered.width > TARGET_W or rendered.height > TARGET_H:
        rendered.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), BG_COLOR)
    canvas.paste(rendered, (0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True, compress_level=6)
    return "ok"


def render_worker(args):
    code_text, out_path_str = args
    return render_code_to_png(code_text, Path(out_path_str))


# ── Message builders (match finetune/00_prepare_data_v2.py) ────────────────
SYSTEM_MSG = (
    "You are a code analysis assistant. "
    "Determine functional equivalence of code snippets across programming languages."
)
INSTRUCTION = (
    "Determine whether the following two code snippets implement the same "
    "functionality, regardless of the programming language used. "
    'Answer with only "Yes" or "No".'
)


def make_b1_messages(lang1, code1, lang2, code2, label):
    user_content = (
        f"{INSTRUCTION}\n\n"
        f"Code 1 (language: {lang1}):\n"
        f"```{lang1.lower()}\n{code1}\n```\n\n"
        f"Code 2 (language: {lang2}):\n"
        f"```{lang2.lower()}\n{code2}\n```\n\n"
        "Answer:"
    )
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": "Yes" if label == 1 else "No"},
    ]


def make_b2_messages(lang1, code1, lang2, code2, label):
    user_content = [
        {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: {lang1}):"},
        {"type": "image"},
        {"type": "text",  "text": f"```{lang1.lower()}\n{code1}\n```\n\nCode 2 (language: {lang2}):"},
        {"type": "image"},
        {"type": "text",  "text": f"```{lang2.lower()}\n{code2}\n```\n\nAnswer:"},
    ]
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": "Yes" if label == 1 else "No"},
    ]


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test set ──────────────────────────────────────────────────────
    print("Loading POJ-104 test set...")
    records = []
    with open(POJ_DIR / "test.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records)} programs, "
          f"{len(set(r['label'] for r in records))} problems")

    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    labels = sorted(by_label.keys())

    # ── Sample pairs ───────────────────────────────────────────────────────
    print(f"\nSampling {N_POS} positive + {N_NEG} negative pairs...")
    pos_pairs, neg_pairs = [], []

    # positive: same label, two distinct programs
    for _ in range(N_POS * 10):
        lbl = random.choice(labels)
        a, b = random.sample(by_label[lbl], 2)
        pos_pairs.append((a, b, 1))
        if len(pos_pairs) == N_POS:
            break

    # negative: different labels
    for _ in range(N_NEG * 10):
        l1, l2 = random.sample(labels, 2)
        a = random.choice(by_label[l1])
        b = random.choice(by_label[l2])
        neg_pairs.append((a, b, 0))
        if len(neg_pairs) == N_NEG:
            break

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    print(f"  Generated {len(all_pairs)} pairs "
          f"(pos={sum(1 for _,_,l in all_pairs if l==1)}, "
          f"neg={sum(1 for _,_,l in all_pairs if l==0)})")

    # ── Collect unique programs to render ─────────────────────────────────
    seen = {}
    for (a, b, _) in all_pairs:
        for r in (a, b):
            key = r["index"]
            if key not in seen:
                seen[key] = r

    print(f"\nRendering {len(seen)} unique C code images...")
    render_tasks = []
    for idx, r in seen.items():
        out_path = IMAGES_DIR / r["label"] / f"{r['index']}.png"
        render_tasks.append((r["code"], str(out_path)))

    counters = {"ok": 0, "skip": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for status in tqdm(pool.map(render_worker, render_tasks,
                                    chunksize=50), total=len(render_tasks)):
            if status == "ok":
                counters["ok"] += 1
            elif status == "skip":
                counters["skip"] += 1
            else:
                counters["error"] += 1
    print(f"  ok={counters['ok']}  skip={counters['skip']}  "
          f"error={counters['error']}")

    # ── Write JSONL files ──────────────────────────────────────────────────
    b1_path = OUT_DIR / "test_b1.jsonl"
    b2_path = OUT_DIR / "test_b2.jsonl"

    print(f"\nWriting {b1_path.name} and {b2_path.name}...")
    with open(b1_path, "w") as fb1, open(b2_path, "w") as fb2:
        for i, (a, b, label) in enumerate(all_pairs):
            lang = "C"
            img_a = f"poj104/images/{a['label']}/{a['index']}.png"
            img_b = f"poj104/images/{b['label']}/{b['index']}.png"

            rec_b1 = {
                "pair_id":     f"poj_{i:05d}",
                "label":       label,
                "neg_type":    None,
                "answer":      "Yes" if label == 1 else "No",
                "lang_pair":   "C-C",
                "messages":    make_b1_messages(lang, a["code"], lang, b["code"], label),
                "image_paths": None,
            }
            rec_b2 = {**rec_b1,
                      "messages":    make_b2_messages(lang, a["code"], lang, b["code"], label),
                      "image_paths": [img_a, img_b]}

            fb1.write(json.dumps(rec_b1) + "\n")
            fb2.write(json.dumps(rec_b2) + "\n")

    print(f"  Wrote {len(all_pairs)} pairs each to B1 and B2 files.")
    print(f"\nDone. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
