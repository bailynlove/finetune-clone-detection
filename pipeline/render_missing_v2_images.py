#!/usr/bin/env python3
"""
Render missing images referenced by splits_v2 train_a / val / test_sd / test_dd
(only AC images that v1 never rendered). Reuses the v2 rendering style.
"""
import json
import re
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.style import Style
from pygments.token import Token

DATASET = Path("/data1/clone-test/dataset")
CODENET = Path("/data1/clone-test/Project_CodeNet")
SPLITS  = DATASET / "splits_v2"

TARGET_W, TARGET_H = 1440, 896
BG_COLOR   = "#272822"
IMAGE_PAD  = 10
MAX_LINES  = 74
MAX_CHARS  = 148
FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

LANG_TO_LEXER = {
    "Python": "python", "Java": "java",
    "Rust":   "rust",   "Ruby": "ruby",
    "C++":    "cpp",
}

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
    code = re.sub(r"[^\x00-\x7F]", " ", raw)
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


def main():
    # Collect all (src, dst, lang) for missing images across all v2 splits
    tasks_set = {}
    splits_files = sorted(SPLITS.rglob("*.jsonl"))
    print(f"Scanning {len(splits_files)} JSONL files in {SPLITS}…")
    for jf in splits_files:
        with open(jf) as f:
            for line in f:
                r = json.loads(line)
                for ck in ['code1', 'code2']:
                    c = r[ck]
                    dst = DATASET / c['image_rel_path']
                    if not dst.exists():
                        src = CODENET / c['rel_path']
                        if src.exists():
                            tasks_set[str(dst)] = (str(src), str(dst), c['lang'])

    tasks = list(tasks_set.values())
    print(f"Found {len(tasks):,} missing images to render (source available)")

    ok = err = skip = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(render_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            st = fut.result()
            if st == "ok":      ok += 1
            elif st == "exists": skip += 1
            else:               err += 1
            done += 1
            if done % 500 == 0:
                print(f"  {done:,}/{len(tasks):,}  ok={ok}  err={err}  skip={skip}")
    print(f"\nDone: ok={ok}  err={err}  skip={skip}")


if __name__ == "__main__":
    main()
