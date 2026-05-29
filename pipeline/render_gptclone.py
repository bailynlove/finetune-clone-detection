#!/usr/bin/env python3
"""Render GPTCloneBench code snippets to 1440×896 PNG images."""
import re
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import ImageFormatter
from pygments.style import Style
from pygments.token import Token

CODE_DIR   = Path("/data1/clone-test/dataset/gptclonebench_code")
IMAGES_DIR = Path("/data1/clone-test/dataset/images/gptclonebench")
TARGET_W, TARGET_H = 1440, 896
IMAGE_PAD  = 10
BG_COLOR   = "#272822"
MAX_LINES  = 74
MAX_CHARS  = 100
FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

LANG_TO_LEXER = {"java": "java", "python": "python"}

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

def render_file(src_path, out_path, lang_key):
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path), "exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = Path(src_path).read_text(errors="replace")
    except Exception as e:
        return str(out_path), f"read_error:{e}"
    code = re.sub(r'[^\x00-\x7F]', ' ', raw)
    lines = code.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES-1] + ["# ... <truncated>"]
    if not lines:
        return str(out_path), "empty"
    code = "\n".join(lines)
    n_lines = len(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)
    font = pick_font(n_lines, max_chars)
    line_pad = max(2, int(font * 0.2))
    lexer_name = LANG_TO_LEXER.get(lang_key, "text")
    try:
        lexer = get_lexer_by_name(lexer_name, stripall=True)
    except Exception:
        lexer = TextLexer()
    fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=font,
                         style=UnifiedMonokaiStyle, line_numbers=False,
                         image_pad=IMAGE_PAD, line_pad=line_pad)
    try:
        rendered = Image.open(BytesIO(highlight(code, lexer, fmt))).convert("RGB")
    except Exception as e:
        return str(out_path), f"render_error:{e}"
    if rendered.width > TARGET_W or rendered.height > TARGET_H:
        rendered.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), BG_COLOR)
    canvas.paste(rendered, (0, 0))
    canvas.save(out_path, "PNG")
    return str(out_path), "ok"

def main():
    tasks = []
    for lang_key, ext in [("java", ".java"), ("python", ".py")]:
        src_dir = CODE_DIR / lang_key
        out_dir = IMAGES_DIR / lang_key
        for src in sorted(src_dir.glob(f"*{ext}")):
            out = out_dir / (src.stem + ".png")
            tasks.append((str(src), str(out), lang_key))
    
    print(f"Rendering {len(tasks)} images...")
    ok = err = skip = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(render_file, s, o, l): (s, o) for s, o, l in tasks}
        for i, fut in enumerate(as_completed(futs)):
            _, status = fut.result()
            if status == "ok": ok += 1
            elif status == "exists": skip += 1
            else: err += 1
            if (i+1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)} done (ok={ok} err={err} skip={skip})")
    print(f"Done: ok={ok} err={err} skip={skip}")

if __name__ == "__main__":
    main()
