#!/usr/bin/env python3
"""
Build GPTCloneBench B1 + B2 eval sets for 4 new language pairs:
  Java <-> C++, Python <-> C++, Python <-> C#, Java <-> C#

Source: cross_language.zip  (java_to_other + cs_to_other validate files)
Each pair: up to 1000 positive + 1000 negative, both B1 (text) and B2 (image+text).
"""
import json, random, zipfile
from pathlib import Path
from collections import defaultdict

random.seed(42)

ROOT      = Path("/data1/clone-test")
ZIP_PATH  = ROOT / "GPTCloneBench/cross_language.zip"
FT_DIR    = ROOT / "dataset/finetune_data"
CODE_DIR  = ROOT / "dataset/gptclonebench_code"
IMAGES    = ROOT / "dataset/images/gptclonebench"

SEP  = '$' * 40
DASH = '-' * 40

SYSTEM_MSG  = ("You are a code analysis assistant. "
               "Determine functional equivalence of code snippets across programming languages.")
INSTRUCTION = ("Determine whether the following two code snippets implement the same "
               "functionality, regardless of the programming language used. "
               'Answer with only "Yes" or "No".')

LANG_DISPLAY = {"java": "Java", "python": "Python", "cpp": "C++", "csharp": "C#"}
LANG_MD      = {"java": "java", "python": "python", "cpp": "cpp", "csharp": "csharp"}
LANG_EXT     = {"java": ".java", "python": ".py", "cpp": ".cpp", "csharp": ".cs"}

TARGET_PAIRS = [
    ("java",   "cpp"),
    ("python", "cpp"),
    ("python", "csharp"),
    ("java",   "csharp"),
]


def detect_lang(code: str) -> str | None:
    c = code.strip()
    if not c:
        return None

    # Python: def + colon without braces as control flow
    if "def " in c and ":" in c and c.count("{") < c.count("def "):
        return "python"
    if c.startswith("def ") or ("self." in c and "def " in c):
        return "python"
    if ("elif " in c or "    pass" in c) and "{" not in c:
        return "python"

    # C++: pointer ops, scope resolution, includes, streams
    if "#include" in c or "std::" in c or "cout <<" in c or "cin >>" in c:
        return "cpp"
    if "::" in c and "System." not in c and "java." not in c:
        return "cpp"
    if "->" in c and "System." not in c and "java." not in c:
        return "cpp"

    # C#: .NET namespaces, Console, var+foreach, C#-specific keywords
    if "System.Windows" in c or "System.Collections" in c or "System.IO" in c:
        return "csharp"
    if "Console.WriteLine" in c or "Console.Write(" in c or "Console.Read" in c:
        return "csharp"
    if "using System" in c or "namespace " in c:
        return "csharp"
    if "var " in c and "foreach" in c:
        return "csharp"
    if " bool " in c and "System." in c and "java." not in c:
        return "csharp"

    # Java: java.* imports, System.out, boolean keyword, collections
    if "System.out" in c or "java.util" in c or "java.awt" in c or "import java" in c:
        return "java"
    if "ArrayList<" in c or "HashMap<" in c or "LinkedList<" in c:
        return "java"
    if "public boolean" in c or "public int" in c or "public void" in c:
        if "System." not in c or "System.out" in c:
            return "java"
    if ("public static" in c or "public class" in c) and "Console." not in c:
        return "java"

    return None


def parse_all_pairs(zip_path: Path) -> dict[tuple, list]:
    buckets: dict[tuple, list] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as z:
        for txt_name in [
            "java_to_other/1_java_to_other_validateClones.txt",
            "cs_to_other/cs_to_other_validateClones.txt",
        ]:
            txt = z.read(txt_name).decode(errors="replace")
            for block in txt.split(SEP):
                block = block.strip()
                if not block:
                    continue
                parts = block.split(DASH)
                if len(parts) < 3:
                    continue
                c1, c2 = parts[1].strip(), parts[2].strip()
                if not c1 or not c2:
                    continue
                l1 = detect_lang(c1)
                l2 = detect_lang(c2)
                if l1 is None or l2 is None or l1 == l2:
                    continue
                key = tuple(sorted([l1, l2]))
                buckets[key].append({l1: c1, l2: c2})
    return buckets


def build_negatives(pos_list, lang_a, lang_b, n):
    negs = []
    used = set()
    for i in range(len(pos_list)):
        for j in range(len(pos_list)):
            if i != j and (i, j) not in used:
                negs.append({lang_a: pos_list[i][lang_a], lang_b: pos_list[j][lang_b]})
                used.add((i, j))
                if len(negs) >= n:
                    return negs
    return negs


print("Parsing cross_language.zip …")
buckets = parse_all_pairs(ZIP_PATH)
for k, v in sorted(buckets.items()):
    print(f"  {k[0]} <-> {k[1]}: {len(v)} positive pairs")


for lang_a, lang_b in TARGET_PAIRS:
    key = tuple(sorted([lang_a, lang_b]))
    pairs = buckets.get(key, [])
    if not pairs:
        print(f"\n[SKIP] No pairs for {lang_a} <-> {lang_b}")
        continue

    random.seed(42)
    random.shuffle(pairs)
    pos_sample = pairs[:1000]
    neg_sample = build_negatives(pos_sample, lang_a, lang_b, 1000)

    all_records = []
    for i, p in enumerate(pos_sample):
        all_records.append({"label": 1, "idx": i, **p})
    for i, p in enumerate(neg_sample):
        all_records.append({"label": 0, "idx": i, **p})

    random.seed(123)
    random.shuffle(all_records)
    random.seed(42)

    pair_name = f"gptclone_{lang_a}_{lang_b}"
    da, db   = LANG_DISPLAY[lang_a], LANG_DISPLAY[lang_b]
    ma, mb   = LANG_MD[lang_a],      LANG_MD[lang_b]
    ea, eb   = LANG_EXT[lang_a],     LANG_EXT[lang_b]

    # Write code files to disk (for image rendering)
    code_dir_a = CODE_DIR / lang_a
    code_dir_b = CODE_DIR / lang_b
    code_dir_a.mkdir(parents=True, exist_ok=True)
    code_dir_b.mkdir(parents=True, exist_ok=True)

    enriched = []
    for r in all_records:
        label  = r["label"]
        tag    = "pos" if label == 1 else "neg"
        pid    = f"gptclone_{lang_a}_{lang_b}_{tag}_{r['idx']:06d}"
        code_a = r[lang_a]
        code_b = r[lang_b]

        fa = code_dir_a / f"{pid}_a{ea}"
        fb = code_dir_b / f"{pid}_b{eb}"
        if not fa.exists():
            fa.write_text(code_a, encoding="utf-8")
        if not fb.exists():
            fb.write_text(code_b, encoding="utf-8")

        enriched.append({
            "pair_id":    pid,
            "label":      label,
            "lang_a":     lang_a,
            "lang_b":     lang_b,
            "code_a":     code_a,
            "code_b":     code_b,
            "img_rel_a":  f"gptclonebench/{lang_a}/{pid}_a.png",
            "img_rel_b":  f"gptclonebench/{lang_b}/{pid}_b.png",
        })

    # B1 (text-only)
    b1_out = FT_DIR / "b1" / pair_name / "test.jsonl"
    b1_out.parent.mkdir(parents=True, exist_ok=True)
    with open(b1_out, "w") as f:
        for r in enriched:
            user_content = (
                f"{INSTRUCTION}\n\n"
                f"Code 1 (language: {da}):\n```{ma}\n{r['code_a']}\n```\n\n"
                f"Code 2 (language: {db}):\n```{mb}\n{r['code_b']}\n```\n\nAnswer:"
            )
            f.write(json.dumps({
                "pair_id":     r["pair_id"],
                "label":       r["label"],
                "answer":      "Yes" if r["label"] == 1 else "No",
                "lang_pair":   f"{da}-{db}",
                "messages": [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": "Yes" if r["label"] == 1 else "No"},
                ],
                "image_paths": None,
            }) + "\n")

    # B2 (image + text)
    b2_out = FT_DIR / "b2" / pair_name / "test.jsonl"
    b2_out.parent.mkdir(parents=True, exist_ok=True)
    with open(b2_out, "w") as f:
        for r in enriched:
            user_content = [
                {"type": "text",  "text": f"{INSTRUCTION}\n\nCode 1 (language: {da}):"},
                {"type": "image"},
                {"type": "text",  "text": f"```{ma}\n{r['code_a']}\n```\n\nCode 2 (language: {db}):"},
                {"type": "image"},
                {"type": "text",  "text": f"```{mb}\n{r['code_b']}\n```\n\nAnswer:"},
            ]
            f.write(json.dumps({
                "pair_id":     r["pair_id"],
                "label":       r["label"],
                "answer":      "Yes" if r["label"] == 1 else "No",
                "lang_pair":   f"{da}-{db}",
                "messages": [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": "Yes" if r["label"] == 1 else "No"},
                ],
                "image_paths": [r["img_rel_a"], r["img_rel_b"]],
            }) + "\n")

    n_pos = sum(1 for r in enriched if r["label"] == 1)
    n_neg = sum(1 for r in enriched if r["label"] == 0)
    print(f"\n[{pair_name}]  pos={n_pos}  neg={n_neg}  total={n_pos+n_neg}")
    print(f"  B1: {b1_out}")
    print(f"  B2: {b2_out}")

print("\nCode files written. Run render_gptclone_extra.py to generate images.")
