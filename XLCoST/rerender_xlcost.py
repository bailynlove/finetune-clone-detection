#!/usr/bin/env python3
"""
Re-render XLCoST images with improved detokenization (remove excess spaces),
and update B1/B2 JSONL files with cleaned code text.

Run this, then re-run 3.5-B2 eval.
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

ROOT       = Path("/data1/clone-test")
FT_DIR     = ROOT / "dataset/finetune_data"
IMAGES_DIR = ROOT / "dataset/images/xlcost"

TARGET_W, TARGET_H = 1440, 896
BG_COLOR   = "#272822"
IMAGE_PAD  = 10
MAX_LINES  = 74
MAX_CHARS  = 100
FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

SYSTEM_MSG  = ("You are a code analysis assistant. "
               "Determine functional equivalence of code snippets across programming languages.")
INSTRUCTION = ("Determine whether the following two code snippets implement the same "
               "functionality, regardless of the programming language used. "
               'Answer with only "Yes" or "No".')

LANG_INFO = {
    "Java":   ("Java",   "java",   "java"),
    "Python": ("Python", "python", "python"),
    "C++":    ("C++",    "cpp",    "cpp"),
    "C#":     ("C#",     "csharp", "csharp"),
}

# ── Code cleanup ───────────────────────────────────────────────────────────────

def cleanup_code(code: str, lang: str) -> str:
    """Remove XLCoST tokenization spacing artifacts from detokenized code."""
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        # Remove space before closing brackets, semicolons, commas
        line = re.sub(r" (?=[)\];,])", "", line)
        # Remove space after opening brackets
        line = re.sub(r"(?<=[(\[]) ", "", line)
        # Remove space before ( and [ when preceded by identifier/closing-bracket
        # (method calls, array access): foo (, arr [  →  foo(, arr[
        line = re.sub(r"(\w) \(", r"\1(", line)
        line = re.sub(r"(\)) \(", r"\1(", line)
        line = re.sub(r"(\w) \[", r"\1[", line)
        line = re.sub(r"(\]) \[", r"\1[", line)
        # Remove space around dot for member/package access
        line = re.sub(r"(\w) \. (\w)", r"\1.\2", line)
        line = re.sub(r"(\w) \. \*", r"\1.*", line)   # java.util.*
        # Compound operators
        # Post-increment/decrement: i ++ → i++, i -- → i--
        line = re.sub(r"(\w) \+\+", r"\1++", line)
        line = re.sub(r"(\w) --", r"\1--", line)
        for pat, rep in [
            (r"\+ \+",  "++"),
            (r"- -",    "--"),
            (r"\+ =",   "+="),
            (r"- =",    "-="),
            (r"\* =",   "*="),
            (r"/ =",    "/="),
            (r"% =",    "%="),
            (r"& =",    "&="),
            (r"\| =",   "|="),
            (r"\^ =",   "^="),
            (r"& &",    "&&"),
            (r"\| \|",  "||"),
            (r"! =",    "!="),
            (r"= =",    "=="),
            (r"< =",    "<="),
            (r"> =",    ">="),
            (r"< <",    "<<"),
            (r"> >",    ">>"),
            (r"- >",    "->"),
            (r": :",    "::"),
        ]:
            line = re.sub(pat, rep, line)
        # Collapse empty braces/brackets: { } → {}
        line = re.sub(r"\{ \}", "{}", line)
        line = re.sub(r"\[ \]", "[]", line)
        if lang == "Python":
            # Remove space before colon (end-of-line control flow + inline dicts/slices)
            line = re.sub(r" :", ":", line)
        cleaned.append(line)
    return "\n".join(cleaned)


_CODE_RE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
_LANG_HEADER_RE = re.compile(r"Code \d+ \(language: ([^)]+)\)")


def extract_codes(messages):
    """Extract (lang_a, code_a, lang_b, code_b, lex_a, lex_b) from messages."""
    user_msg = next(m for m in messages if m["role"] == "user")
    content = user_msg["content"]

    if isinstance(content, str):
        text = content
    else:
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")

    langs_disp = _LANG_HEADER_RE.findall(text)
    code_blocks = _CODE_RE.findall(text)   # list of (lexer, code)

    if len(langs_disp) < 2 or len(code_blocks) < 2:
        return None

    lang_a, lang_b = langs_disp[0], langs_disp[1]
    _, code_a = code_blocks[0]
    _, code_b = code_blocks[1]
    return lang_a, code_a, lang_b, code_b


# ── Image rendering ────────────────────────────────────────────────────────────

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


def render_png(code, lexer_name, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    code = re.sub(r"[^\x00-\x7F]", " ", code)
    lines = code.splitlines()
    if not lines:
        return "empty"
    n_lines   = len(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)
    font      = pick_font(n_lines, max_chars)
    line_pad  = max(2, int(font * 0.2))
    try:
        lexer = get_lexer_by_name(lexer_name, stripall=True)
    except Exception:
        lexer = TextLexer()
    fmt = ImageFormatter(
        font_name="DejaVu Sans Mono", font_size=font,
        style=UnifiedMonokaiStyle, line_numbers=False,
        image_pad=IMAGE_PAD, line_pad=line_pad,
    )
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


def _render_task(args):
    return render_png(*args)


# ── Main ───────────────────────────────────────────────────────────────────────

PAIRS = [
    ("xlcost_java_python",   "Java",   "Python"),
    ("xlcost_java_cpp",      "Java",   "C++"),
    ("xlcost_java_csharp",   "Java",   "C#"),
    ("xlcost_cpp_python",    "C++",    "Python"),
    ("xlcost_cpp_csharp",    "C++",    "C#"),
    ("xlcost_python_csharp", "Python", "C#"),
]


def rebuild_pair(pair_name, lang_a_disp, lang_b_disp):
    _, mda, lexa = LANG_INFO[lang_a_disp]
    _, mdb, lexb = LANG_INFO[lang_b_disp]
    subdir_a = LANG_INFO[lang_a_disp][2]
    subdir_b = LANG_INFO[lang_b_disp][2]

    b1_in  = FT_DIR / "b1" / pair_name / "test.jsonl"
    b2_in  = FT_DIR / "b2" / pair_name / "test.jsonl"

    if not b1_in.exists():
        print(f"  [SKIP] {pair_name} — no B1 JSONL")
        return

    # Read B1 (source of truth for code text — simpler to parse)
    records_b1 = [json.loads(l) for l in b1_in.read_text().splitlines()]

    # Parse code from B1 messages, apply cleanup, rebuild both B1 & B2 JSONL + images
    render_tasks = []
    new_b1, new_b2 = [], []

    for r in records_b1:
        extracted = extract_codes(r["messages"])
        if extracted is None:
            new_b1.append(r)
            new_b2.append(json.loads((FT_DIR/"b2"/pair_name/"test.jsonl").read_text().split("\n")[records_b1.index(r)]))
            continue

        lang_a, code_a_raw, lang_b, code_b_raw = extracted
        code_a = cleanup_code(code_a_raw, lang_a)
        code_b = cleanup_code(code_b_raw, lang_b)

        img_rel_a = f"images/xlcost/{subdir_a}/{r['pair_id']}_a.png"
        img_rel_b = f"images/xlcost/{subdir_b}/{r['pair_id']}_b.png"
        img_abs_a = ROOT / "dataset" / img_rel_a
        img_abs_b = ROOT / "dataset" / img_rel_b

        render_tasks.append((code_a, lexa, img_abs_a))
        render_tasks.append((code_b, lexb, img_abs_b))

        # B1 record
        user_b1 = (
            f"{INSTRUCTION}\n\n"
            f"Code 1 (language: {lang_a}):\n```{mda}\n{code_a}\n```\n\n"
            f"Code 2 (language: {lang_b}):\n```{mdb}\n{code_b}\n```\n\nAnswer:"
        )
        new_b1.append({**r, "messages": [
            r["messages"][0],
            {"role": "user",      "content": user_b1},
            r["messages"][2],
        ]})

        # B2 record
        user_b2 = [
            {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: {lang_a}):"},
            {"type": "image"},
            {"type": "text",  "text": f"```{mda}\n{code_a}\n```\n\nCode 2 (language: {lang_b}):"},
            {"type": "image"},
            {"type": "text",  "text": f"```{mdb}\n{code_b}\n```\n\nAnswer:"},
        ]
        b2_base = {k: v for k, v in r.items() if k != "image_paths"}
        new_b2.append({**b2_base, "messages": [
            r["messages"][0],
            {"role": "user",      "content": user_b2},
            r["messages"][2],
        ], "image_paths": [img_rel_a, img_rel_b]})

    # Re-render images (force overwrite)
    print(f"  [{pair_name}] rendering {len(render_tasks)} images…")
    ok = err = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_render_task, t): t for t in render_tasks}
        for fut in as_completed(futs):
            st = fut.result()
            if st == "ok": ok += 1
            else:          err += 1; print(f"    WARN: {st}")
    print(f"    ok={ok}  err={err}")

    # Write updated JSONL
    b1_in.write_text("\n".join(json.dumps(r) for r in new_b1) + "\n")
    b2_in.write_text("\n".join(json.dumps(r) for r in new_b2) + "\n")
    print(f"    JSONL updated: {b1_in.name}  {b2_in.name}")


def main():
    print("Re-rendering XLCoST images with cleaned code…")
    for pair_name, lang_a, lang_b in PAIRS:
        rebuild_pair(pair_name, lang_a, lang_b)
    print("\nDone. Now re-run 3.5-B2 eval.")


if __name__ == "__main__":
    main()
