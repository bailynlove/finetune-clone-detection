#!/usr/bin/env python3
"""
Build XLCoST clone-detection test sets (B1 + B2) following v2 scheme.

XLCoST contains parallel aligned program pairs (same algorithm, different language).
All aligned pairs are semantically equivalent → label=1 (POS).
Cross-pair samples → label=0 (EASY_NEG).  No WA submissions → no HARD_NEG.

Mix: 50% POS + 50% EASY_NEG  (HARD_NEG unavailable)
Size: 500 POS + 500 EASY_NEG = 1000 per language pair

Language pairs built:
  Java↔Python, Java↔C++, Java↔C#, C++↔Python, C++↔C#, Python↔C#
"""
import json
import random
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

random.seed(42)

ROOT         = Path("/data1/clone-test")
XLCOST_DIR   = ROOT / "XLCoST/XLCoST_data/generation/pair_data_tok_full"
FT_DIR       = ROOT / "dataset/finetune_data"
IMAGES_DIR   = ROOT / "dataset/images/xlcost"

TARGET_W, TARGET_H = 1440, 896
BG_COLOR   = "#272822"
IMAGE_PAD  = 10
MAX_LINES  = 74
MAX_CHARS  = 100
FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

N_POS = 500
N_NEG = 500

SYSTEM_MSG  = ("You are a code analysis assistant. "
               "Determine functional equivalence of code snippets across programming languages.")
INSTRUCTION = ("Determine whether the following two code snippets implement the same "
               "functionality, regardless of the programming language used. "
               'Answer with only "Yes" or "No".')

# XLCoST dir name → (display_name, md_name, file_ext, lexer_name, image_subdir)
LANG_INFO = {
    "Java":       ("Java",       "java",   ".java", "java",   "java"),
    "Python":     ("Python",     "python", ".py",   "python", "python"),
    "C++":        ("C++",        "cpp",    ".cpp",  "cpp",    "cpp"),
    "C#":         ("C#",         "csharp", ".cs",   "csharp", "csharp"),
}

TARGET_PAIRS = [
    ("Java",   "Python"),
    ("Java",   "C++"),
    ("Java",   "C#"),
    ("C++",    "Python"),
    ("C++",    "C#"),
    ("Python", "C#"),
]

# ── Detokenizers ──────────────────────────────────────────────────────────────

def detok_python(s: str) -> list[str]:
    """XLCoST Python uses NEW_LINE / INDENT / DEDENT special tokens."""
    tokens = s.split()
    lines, indent, cur = [], 0, []
    for t in tokens:
        if t == "NEW_LINE":
            lines.append("    " * indent + " ".join(cur).strip())
            cur = []
        elif t == "INDENT":
            indent += 1
        elif t == "DEDENT":
            indent = max(0, indent - 1)
        else:
            cur.append(t)
    if cur:
        lines.append("    " * indent + " ".join(cur).strip())
    return [l for l in lines if l.strip()]


def detok_cstyle(s: str) -> list[str]:
    """XLCoST C-style (Java / C++ / C#): braces and semicolons delimit lines."""
    tokens = s.split()
    lines, indent, cur = [], 0, []
    for t in tokens:
        if t == "{":
            cur.append("{")
            lines.append("    " * indent + " ".join(cur).strip())
            cur = []
            indent += 1
        elif t == "}":
            if cur:
                lines.append("    " * indent + " ".join(cur).strip())
                cur = []
            indent = max(0, indent - 1)
            lines.append("    " * indent + "}")
        elif t == ";":
            cur.append(";")
            lines.append("    " * indent + " ".join(cur).strip())
            cur = []
        else:
            cur.append(t)
    if cur:
        lines.append("    " * indent + " ".join(cur).strip())
    return [l for l in lines if l.strip()]


def detok(s: str, lang: str) -> list[str]:
    if lang == "Python":
        return detok_python(s)
    return detok_cstyle(s)


def prepare_code(lines: list[str]) -> str:
    if len(lines) > MAX_LINES:
        lines = lines[: MAX_LINES - 1] + ["// ... <truncated>"]
    return "\n".join(lines)


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


def pick_font(n_lines: int, max_chars: int) -> int:
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


def render_code_to_png(code: str, lexer_name: str, out_path: Path) -> str:
    out_path = Path(out_path)
    if out_path.exists():
        return "exists"
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


def render_task(args):
    code, lexer_name, out_path = args
    return render_code_to_png(code, lexer_name, out_path)


# ── Build one language pair ────────────────────────────────────────────────────

def build_pair(lang_a: str, lang_b: str):
    da, _, _, lexa, subdir_a = LANG_INFO[lang_a]
    db, _, _, lexb, subdir_b = LANG_INFO[lang_b]
    mda = LANG_INFO[lang_a][1]
    mdb = LANG_INFO[lang_b][1]

    pair_dir = XLCOST_DIR / f"{lang_a}-{lang_b}"
    if not pair_dir.exists():
        # Try reversed
        pair_dir = XLCOST_DIR / f"{lang_b}-{lang_a}"
        if not pair_dir.exists():
            print(f"[SKIP] No dir for {lang_a}-{lang_b}")
            return
        lang_a, lang_b = lang_b, lang_a
        da, db = db, da
        mda, mdb = mdb, mda
        lexa, lexb = lexb, lexa
        subdir_a, subdir_b = subdir_b, subdir_a

    # Find test files
    files_a = sorted(pair_dir.glob(f"test-{lang_a}-{lang_b}-tok.*"))
    files_b = sorted(pair_dir.glob(f"test-{lang_b}-{lang_a}-tok.*"))
    if not files_a:
        files_a = [f for f in pair_dir.iterdir()
                   if f.name.startswith("test-") and
                   lang_a.lower() in f.name.lower() and
                   lang_b.lower() in f.name.lower()]
    # Each pair dir has exactly 2 aligned test files: one per language
    test_files = sorted(pair_dir.glob("test-*-tok.*"))
    if len(test_files) != 2:
        print(f"[WARN] Expected 2 test files for {lang_a}-{lang_b}, found {len(test_files)}")
        return

    # Identify which file belongs to which language by extension
    ext_a = LANG_INFO[lang_a][2]
    ext_b = LANG_INFO[lang_b][2]
    file_a = next((f for f in test_files if f.suffix == ext_a), None)
    file_b = next((f for f in test_files if f.suffix == ext_b), None)
    if file_a is None or file_b is None:
        print(f"[SKIP] Can't match files by extension for {lang_a}-{lang_b}: "
              f"{[f.name for f in test_files]}")
        return

    lines_a = file_a.read_text(errors="replace").splitlines()
    lines_b = file_b.read_text(errors="replace").splitlines()
    n = min(len(lines_a), len(lines_b))
    print(f"  [{lang_a} ↔ {lang_b}] {n} aligned test pairs")

    codes_a = [prepare_code(detok(l, lang_a)) for l in lines_a[:n]]
    codes_b = [prepare_code(detok(l, lang_b)) for l in lines_b[:n]]

    rng = random.Random(42)
    indices = list(range(n))
    rng.shuffle(indices)

    # POS: same-index pairs
    pos_idx = indices[:N_POS]

    # EASY_NEG: different-index pairs (guaranteed by cycling offset)
    neg_idx_a, neg_idx_b = [], []
    offset = n // 3
    for i in indices[:N_NEG]:
        j = (i + offset) % n
        if j == i:
            j = (i + 1) % n
        neg_idx_a.append(i)
        neg_idx_b.append(j)

    pair_name = f"xlcost_{lang_a.lower().replace('+','p').replace('#','sharp')}_{lang_b.lower().replace('+','p').replace('#','sharp')}"
    img_subdir_a = IMAGES_DIR / subdir_a
    img_subdir_b = IMAGES_DIR / subdir_b
    img_subdir_a.mkdir(parents=True, exist_ok=True)
    img_subdir_b.mkdir(parents=True, exist_ok=True)

    records = []

    for kind, pairs_idx in [("pos", [(i, i) for i in pos_idx]),
                             ("neg", list(zip(neg_idx_a, neg_idx_b)))]:
        label = 1 if kind == "pos" else 0
        for seq, (ia, ib) in enumerate(pairs_idx):
            pid      = f"{pair_name}_{kind}_{seq:06d}"
            code_a   = codes_a[ia]
            code_b   = codes_b[ib]
            img_rel_a = f"images/xlcost/{subdir_a}/{pid}_a.png"
            img_rel_b = f"images/xlcost/{subdir_b}/{pid}_b.png"
            records.append({
                "pair_id":    pid,
                "label":      label,
                "answer":     "Yes" if label else "No",
                "lang_pair":  f"{da}-{db}",
                "code_a":     code_a,
                "code_b":     code_b,
                "lang_a":     lang_a,
                "lang_b":     lang_b,
                "lex_a":      lexa,
                "lex_b":      lexb,
                "img_rel_a":  img_rel_a,
                "img_rel_b":  img_rel_b,
            })

    rng.shuffle(records)

    # ── Write B1 ──────────────────────────────────────────────────────────────
    b1_out = FT_DIR / "b1" / pair_name / "test.jsonl"
    b1_out.parent.mkdir(parents=True, exist_ok=True)
    with open(b1_out, "w") as f:
        for r in records:
            user_content = (
                f"{INSTRUCTION}\n\n"
                f"Code 1 (language: {r['lang_a']}):\n```{mda}\n{r['code_a']}\n```\n\n"
                f"Code 2 (language: {r['lang_b']}):\n```{mdb}\n{r['code_b']}\n```\n\nAnswer:"
            )
            f.write(json.dumps({
                "pair_id":     r["pair_id"],
                "label":       r["label"],
                "answer":      r["answer"],
                "lang_pair":   r["lang_pair"],
                "messages": [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": r["answer"]},
                ],
                "image_paths": None,
            }) + "\n")

    # ── Write B2 ──────────────────────────────────────────────────────────────
    b2_out = FT_DIR / "b2" / pair_name / "test.jsonl"
    b2_out.parent.mkdir(parents=True, exist_ok=True)
    with open(b2_out, "w") as f:
        for r in records:
            user_content = [
                {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: {r['lang_a']}):"},
                {"type": "image"},
                {"type": "text",  "text": f"```{mda}\n{r['code_a']}\n```\n\nCode 2 (language: {r['lang_b']}):"},
                {"type": "image"},
                {"type": "text",  "text": f"```{mdb}\n{r['code_b']}\n```\n\nAnswer:"},
            ]
            f.write(json.dumps({
                "pair_id":     r["pair_id"],
                "label":       r["label"],
                "answer":      r["answer"],
                "lang_pair":   r["lang_pair"],
                "messages": [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": r["answer"]},
                ],
                "image_paths": [r["img_rel_a"], r["img_rel_b"]],
            }) + "\n")

    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    print(f"  → B1: {b1_out}  B2: {b2_out}")
    print(f"     pos={n_pos}  neg={n_neg}  total={n_pos+n_neg}")

    # ── Render images ─────────────────────────────────────────────────────────
    render_tasks = []
    for r in records:
        render_tasks.append((r["code_a"], r["lex_a"],
                             ROOT / "dataset" / r["img_rel_a"]))
        render_tasks.append((r["code_b"], r["lex_b"],
                             ROOT / "dataset" / r["img_rel_b"]))

    ok = err = skip = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(render_task, t): t for t in render_tasks}
        for fut in as_completed(futs):
            st = fut.result()
            if st == "ok":       ok   += 1
            elif st == "exists": skip += 1
            else:                err  += 1
    print(f"     images: ok={ok}  err={err}  skip={skip}")


def main():
    print(f"Building XLCoST eval sets → {FT_DIR}/b1|b2/xlcost_*/test.jsonl")
    for lang_a, lang_b in TARGET_PAIRS:
        build_pair(lang_a, lang_b)
    print("\nDone.")


if __name__ == "__main__":
    main()
