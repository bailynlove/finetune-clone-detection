#!/usr/bin/env python3
"""
02_render_images.py
Render all code snippets as syntax-highlighted PNG images at fixed 1440×896.

Per plan §4 (finetune_plan_v2.md):
  - Canvas 1440×896 (image_factor=32, 32×45 × 32×28 patches)
  - Tiered fonts by line count: ≤15→26pt, 16-30→20pt, 31-45→16pt, >45→13pt+truncate
  - Long lines (>100 chars): drop one font tier
  - Non-ASCII → space (DejaVu lacks CJK glyphs)
  - Unified cross-language token coloring (all Keyword.* same color, etc.)
  - Code pasted top-left on Monokai #272822 background; output always 1440×896
"""

import json
import re
from pathlib import Path
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import ImageFormatter
from pygments.style import Style
from pygments.token import (Token, Keyword, Name, Comment, String,
                             Error, Number, Operator, Punctuation, Generic)

CODENET_DIR = Path("/data1/clone-test/Project_CodeNet")
DATASET_DIR = Path("/data1/clone-test/dataset")
SPLITS_DIR  = DATASET_DIR / "splits"
IMAGES_DIR  = DATASET_DIR / "images"

TARGET_W = 1440
TARGET_H = 896
BG_COLOR = (39, 40, 34)   # #272822 Monokai background
IMAGE_PAD = 24

MAX_LINES = 45   # truncate to this many lines (45th = truncation marker)
MAX_CHARS = 100  # lines longer than this drop one font tier

LANG_TO_LEXER = {
    "Python": "python3",
    "Java":   "java",
    "C++":    "cpp",
    "Ruby":   "ruby",
    "Rust":   "rust",
    "Kotlin": "kotlin",
    "Scala":  "scala",
}

# Font tiers: (max_lines, font_pt)
FONT_TIERS = [
    (15, 26),
    (30, 20),
    (45, 16),
    (99, 13),   # code already truncated to ≤45 lines; this tier just catches edge cases
]


class UnifiedMonokaiStyle(Style):
    """Monokai with unified token colors across languages.

    All Keyword.* subtypes inherit the same pink (#f92672), all String.* inherit
    the same yellow (#e6db74), etc. This prevents lexer differences from breaking
    cross-language visual alignment.
    """
    background_color = "#272822"
    highlight_color  = "#49483e"
    default_style    = ""

    styles = {
        Token:              "#f8f8f2",
        Punctuation:        "#f8f8f2",
        Comment:            "#75715e",   # all comments: grey
        Error:              "#960050 bg:#1e0010",
        Generic.Error:      "#960050 bg:#1e0010",
        Keyword:            "#f92672",   # all keywords: same pink (subtypes inherit)
        Number:             "#ae81ff",   # all numbers: same purple
        Operator:           "#f92672",
        Operator.Word:      "#f92672",
        String:             "#e6db74",   # all strings: same yellow (subtypes inherit)
        String.Escape:      "#ae81ff",
        Name:               "#f8f8f2",
        Name.Attribute:     "#a6e22e",
        Name.Builtin:       "#f8f8f2",
        Name.Class:         "#a6e22e",
        Name.Decorator:     "#a6e22e",
        Name.Exception:     "#a6e22e",
        Name.Function:      "#a6e22e",
        Name.Other:         "#a6e22e",
        Name.Tag:           "#f92672",
        Generic:            "#f8f8f2",
        Generic.Emph:       "#f8f8f2",
        Generic.Output:     "#66d9ef",
        Generic.Prompt:     "#f8f8f2",
        Generic.Strong:     "#f8f8f2",
    }


def pick_font(n_lines, max_chars):
    """Select font size (pt) by line count tier, drop one tier for long lines."""
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


def prepare_code(raw_code):
    """Replace non-ASCII, truncate to MAX_LINES lines. Returns (code, n_lines, max_chars)."""
    code = re.sub(r'[^\x00-\x7F]', ' ', raw_code)
    lines = code.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + ["# ... <truncated>"]
    if not lines:
        return "", 0, 0
    code = "\n".join(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)
    return code, len(lines), max_chars


def render_one(args):
    """Render a single code file → 1440×896 PNG. Returns (out_path_str, status)."""
    pid, lang, sid, ext = args
    out_path = IMAGES_DIR / pid / lang / f"{sid}.png"
    src = CODENET_DIR / "data" / pid / lang / f"{sid}.{ext}"

    try:
        raw = src.read_text(errors="replace")
    except Exception as e:
        return str(out_path), f"read_error:{e}"

    if not raw.strip():
        return str(out_path), "empty"

    code, n_lines, max_chars = prepare_code(raw)
    if n_lines == 0:
        return str(out_path), "empty"

    font = pick_font(n_lines, max_chars)
    line_pad = max(2, int(font * 0.2))

    lexer_name = LANG_TO_LEXER.get(lang, "text")
    try:
        lexer = get_lexer_by_name(lexer_name, stripall=True)
    except Exception:
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
        return str(out_path), f"render_error:{e}"

    if rendered.width > TARGET_W or rendered.height > TARGET_H:
        rendered.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)

    canvas = Image.new("RGB", (TARGET_W, TARGET_H), BG_COLOR)
    canvas.paste(rendered, (0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True, compress_level=6)
    return str(out_path), "ok"


def collect_unique_refs():
    refs = {}
    for jsonl in SPLITS_DIR.rglob("*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                try:
                    pair = json.loads(line)
                except Exception:
                    continue
                for key in ("code1", "code2"):
                    c = pair[key]
                    k = (c["problem_id"], c["lang"], c["submission_id"])
                    if k not in refs:
                        ext = c["rel_path"].rsplit(".", 1)[-1]
                        refs[k] = ext
    return refs


def main():
    print("Collecting unique code references from all JSONL files...")
    refs = collect_unique_refs()
    print(f"  Found {len(refs):,} unique code files to render")
    print(f"  Canvas: {TARGET_W}×{TARGET_H}  "
          f"font_tiers={(', '.join(f'≤{n}→{p}pt' for n, p in FONT_TIERS))}  "
          f"max_chars_before_drop={MAX_CHARS}  max_lines={MAX_LINES}")

    tasks = [(pid, lang, sid, ext) for (pid, lang, sid), ext in refs.items()]

    counts = defaultdict(int)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(render_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Rendering"):
            _, status = fut.result()
            counts[status.split(":")[0]] += 1

    print("\nRendering complete:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v:,}")

    import subprocess, random
    result = subprocess.run(["du", "-sh", str(IMAGES_DIR)], capture_output=True, text=True)
    print(f"\nImages directory size: {result.stdout.strip()}")

    all_imgs = list(IMAGES_DIR.rglob("*.png"))
    sample = random.Random(42).sample(all_imgs, min(200, len(all_imgs)))
    sizes = set()
    for p in sample:
        try:
            sizes.add(Image.open(p).size)
        except Exception:
            pass
    print(f"Unique sizes in sample of {len(sample)}: {len(sizes)}")
    print(f"All {TARGET_W}×{TARGET_H}: {sizes == {(TARGET_W, TARGET_H)}}")


if __name__ == "__main__":
    main()
