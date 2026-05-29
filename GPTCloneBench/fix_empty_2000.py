#!/usr/bin/env python3
"""
Replace 16 empty positive pairs (pos_000558..pos_000971) with non-empty
pairs drawn from all_pos[1000:] (the unused pool).
Then re-renders the missing images and rebuilds the finetune JSONLs.
"""
import json, random, zipfile, re
from pathlib import Path
from io import BytesIO
from PIL import Image
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import ImageFormatter
from pygments.style import Style
from pygments.token import Token

random.seed(42)

ROOT       = Path("/data1/clone-test")
CODE_DIR   = ROOT / "dataset/gptclonebench_code"
IMAGES_DIR = ROOT / "dataset/images/gptclonebench"
OUT_DIR    = ROOT / "dataset/finetune_data"
SPLIT_DIR  = ROOT / "dataset/splits/gptclonebench_java_python_2000"

SEP  = '$' * 40
DASH = '-' * 40

TARGET_W, TARGET_H = 1440, 896
IMAGE_PAD  = 10
BG_COLOR   = "#272822"
MAX_LINES  = 74
MAX_CHARS  = 100
FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

SYSTEM_MSG  = ("You are a code analysis assistant. "
               "Determine functional equivalence of code snippets across programming languages.")
INSTRUCTION = ("Determine whether the following two code snippets implement the same "
               "functionality, regardless of the programming language used. "
               'Answer with only "Yes" or "No".')


class UnifiedMonokaiStyle(Style):
    background_color = BG_COLOR
    highlight_color  = "#49483e"
    styles = {
        Token:                "#f8f8f2",
        Token.Keyword:        "#66d9ef",
        Token.Name.Function:  "#a6e22e",
        Token.Literal.String: "#e6db74",
        Token.Literal.Number: "#ae81ff",
        Token.Comment:        "#75715e italic",
        Token.Operator:       "#f92672",
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


def render_code(code_text, lang_key, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    code = re.sub(r'[^\x00-\x7F]', ' ', code_text)
    lines = code.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + ["# ... <truncated>"]
    code = "\n".join(lines)
    n_lines = len(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)
    font = pick_font(n_lines, max_chars)
    line_pad = max(2, int(font * 0.2))
    try:
        lexer = get_lexer_by_name(lang_key, stripall=True)
    except Exception:
        lexer = TextLexer()
    fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=font,
                         style=UnifiedMonokaiStyle, line_numbers=False,
                         image_pad=IMAGE_PAD, line_pad=line_pad)
    rendered = Image.open(BytesIO(highlight(code, lexer, fmt))).convert("RGB")
    if rendered.width > TARGET_W or rendered.height > TARGET_H:
        rendered.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), BG_COLOR)
    canvas.paste(rendered, (0, 0))
    canvas.save(out_path, "PNG")


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


# --- Identify failed pair indices ---
failed_ids = set()
for p in (CODE_DIR / "java").glob("gptclone_java_python_pos_*.java"):
    if p.stat().st_size == 0:
        idx = int(p.stem.split("_pos_")[1].split("_java")[0])
        failed_ids.add(idx)

print(f"Failed pair indices (empty java): {sorted(failed_ids)}")

# --- Rebuild all_pos with same seed ---
all_pos = parse_pairs(
    ROOT / "GPTCloneBench/cross_language.zip",
    "java_to_other/1_java_to_other_validateClones.txt"
)
random.shuffle(all_pos)

# Draw replacements from all_pos[1000:] — skip already-used indices
already_used = set(range(1000))
replacements = []
for i, p in enumerate(all_pos[1000:], start=1000):
    if p['java'].strip() and p['python'].strip():
        replacements.append((i, p))
    if len(replacements) >= len(failed_ids):
        break

print(f"Replacement pairs found: {len(replacements)} from all_pos[1000+]")

# --- Apply replacements ---
for fail_idx, (src_idx, rep) in zip(sorted(failed_ids), replacements):
    pair_id  = f"gptclone_java_python_pos_{fail_idx:06d}"
    java_f   = CODE_DIR / "java"   / f"{pair_id}_java.java"
    py_f     = CODE_DIR / "python" / f"{pair_id}_py.py"
    java_img = IMAGES_DIR / "java"   / f"{pair_id}_java.png"
    py_img   = IMAGES_DIR / "python" / f"{pair_id}_py.png"

    # Overwrite code files
    java_f.write_text(rep['java'],   encoding='utf-8')
    py_f.write_text(rep['python'], encoding='utf-8')

    # Render images
    render_code(rep['java'],   "java",   java_img)
    render_code(rep['python'], "python", py_img)
    print(f"  Replaced pos_{fail_idx:06d} with all_pos[{src_idx}] → rendered OK")

# --- Rebuild the in-memory records for pos_sample with fixes applied ---
# Re-read the current code files as the ground truth (fixes are already applied)
pos_sample_codes = {}
for i in range(1000):
    pair_id = f"gptclone_java_python_pos_{i:06d}"
    java_f  = CODE_DIR / "java"   / f"{pair_id}_java.java"
    py_f    = CODE_DIR / "python" / f"{pair_id}_py.py"
    pos_sample_codes[i] = {
        'java':   java_f.read_text(encoding='utf-8', errors='replace'),
        'python': py_f.read_text(encoding='utf-8', errors='replace'),
    }

# Neg sample codes (no changes needed)
neg_sample_codes = {}
for i in range(1000):
    pair_id = f"gptclone_java_python_neg_{i:06d}"
    java_f  = CODE_DIR / "java"   / f"{pair_id}_java.java"
    py_f    = CODE_DIR / "python" / f"{pair_id}_py.py"
    neg_sample_codes[i] = {
        'java':   java_f.read_text(encoding='utf-8', errors='replace'),
        'python': py_f.read_text(encoding='utf-8', errors='replace'),
    }

# Rebuild records list (same shuffle seed as original 2000 build)
records = []
for i in range(1000):
    pair_id = f"gptclone_java_python_pos_{i:06d}"
    records.append({
        "pair_id":     pair_id,
        "label":       1,
        "java_code":   pos_sample_codes[i]['java'],
        "python_code": pos_sample_codes[i]['python'],
        "code1": {
            "lang": "Java", "problem_id": "gptclone",
            "submission_id": f"{pair_id}_java",
            "image_rel_path": f"gptclonebench/java/{pair_id}_java.png",
        },
        "code2": {
            "lang": "Python", "problem_id": "gptclone",
            "submission_id": f"{pair_id}_py",
            "image_rel_path": f"gptclonebench/python/{pair_id}_py.png",
        },
    })
for i in range(1000):
    pair_id = f"gptclone_java_python_neg_{i:06d}"
    records.append({
        "pair_id":     pair_id,
        "label":       0,
        "java_code":   neg_sample_codes[i]['java'],
        "python_code": neg_sample_codes[i]['python'],
        "code1": {
            "lang": "Java", "problem_id": "gptclone",
            "submission_id": f"{pair_id}_java",
            "image_rel_path": f"gptclonebench/java/{pair_id}_java.png",
        },
        "code2": {
            "lang": "Python", "problem_id": "gptclone",
            "submission_id": f"{pair_id}_py",
            "image_rel_path": f"gptclonebench/python/{pair_id}_py.png",
        },
    })

random.seed(123)
random.shuffle(records)

# --- Rebuild raw split ---
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
with open(SPLIT_DIR / "test.jsonl", 'w') as f:
    for r in records:
        row = {k: v for k, v in r.items() if k not in ('java_code', 'python_code')}
        f.write(json.dumps(row) + '\n')
print("Raw split rebuilt.")

# --- Rebuild finetune JSONLs for all 4 modes ---
for mode in ['b1', 'b1ctrl', 'b2', 'b3']:
    out = OUT_DIR / mode / "gptclone_java_python_2000" / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for r in records:
            jc, pc, label = r['java_code'], r['python_code'], r['label']
            if mode == 'b1':
                user_content = (
                    f"{INSTRUCTION}\n\n"
                    f"Code 1 (language: Java):\n```java\n{jc}\n```\n\n"
                    f"Code 2 (language: Python):\n```python\n{pc}\n```\n\nAnswer:"
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
                        {"type": "text",  "text": f"```java\n{jc}\n```\n\nCode 2 (language: Python):"},
                        {"type": "image"},
                        {"type": "text",  "text": f"```python\n{pc}\n```\n\nAnswer:"},
                    ]
                messages = [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": "Yes" if label == 1 else "No"},
                ]
                img = ["__black__", "__black__"] if mode == 'b1ctrl' else \
                      [r['code1']['image_rel_path'], r['code2']['image_rel_path']]
            f.write(json.dumps({
                "pair_id":    r['pair_id'],
                "label":      label,
                "answer":     "Yes" if label == 1 else "No",
                "lang_pair":  "Java-Python",
                "messages":   messages,
                "image_paths": img,
            }) + '\n')
    n = sum(1 for _ in open(out))
    print(f"[{mode}] gptclone_java_python_2000/test.jsonl: {n} pairs")

print("\nDone.")
