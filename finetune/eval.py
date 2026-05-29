#!/usr/bin/env python3
"""
eval.py — Inference + F1 evaluation for B1 / B1-ctrl / B2 / B3 checkpoints.

Usage:
  python finetune/eval.py --mode b2 \
      --checkpoint runs/b2_train_a_seed42 \
      --base_model /data1/models/Qwen3.5-4B \
      --test_files dataset/finetune_data/b2/test_sd/python_java.jsonl \
      --output results/b2_train_a_python_java_sd.json
"""

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from transformers import AutoProcessor

DATASET_DIR = Path("/data1/clone-test/dataset")
_BLACK_IMG   = Image.new("RGB", (1440, 896), (0, 0, 0))

_THINK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_BLOCK = re.compile(
    r'Code\s+\d+\s*\([^)]*\)\s*:\s*```\w*\n(.*?)\n```',
    re.DOTALL,
)
_LANG_LABEL = re.compile(r'Code\s+\d+\s+\(language:\s*([^)]+)\)')

# ── ian-Kappa style prompt ────────────────────────────────────────────────────
IANKAPPA_SYSTEM = (
    "You are an expert software developer specializing in cross-language code analysis. "
    "Your task is to determine whether two code snippets in different programming languages "
    "are semantic clones."
)

IANKAPPA_DEFINITION = """\
### Definition of Semantic Clone (Cross-Language)

Two code snippets are semantic clones if they implement the same underlying algorithm \
or computational task and would produce identical outputs for all valid inputs, \
regardless of the programming language used.

**Positive (Clone — answer Yes):**
- Both snippets solve the same computational problem; an online judge would accept \
both as correct solutions to that problem.
- Differences in implementation style, variable names, data structures, or language \
idioms do not affect this judgment.
- Different but equivalent algorithms (e.g., iterative vs. recursive) are still clones \
if they produce the same output for all valid inputs.

**Negative (Not Clone — answer No):**
- One snippet contains a bug causing incorrect output on some inputs — e.g., missing \
edge cases (n=0, empty input), integer overflow from using a too-small data type, \
wrong algorithm logic, or incorrect input parsing.
- The snippets solve fundamentally different computational problems.
- One snippet produces different outputs from the other on at least one valid input.

Note: Code that passes most test cases but fails on specific edge cases or large inputs \
is NOT a clone of a fully correct implementation.\
"""

IANKAPPA_INSTRUCTION = (
    '\n\n### Task\n'
    'Are the two code snippets below semantic clones? '
    'Answer with only "Yes" or "No".'
)
# ─────────────────────────────────────────────────────────────────────────────


def load_image(rel_path):
    if rel_path == "__black__":
        return _BLACK_IMG.copy()
    full = DATASET_DIR / rel_path
    try:
        return Image.open(full).convert("RGB")
    except Exception:
        return Image.new("RGB", (1440, 896), (39, 40, 34))


def load_jsonl(paths):
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def _assemble_user_text(content):
    """Return plain text from B1 string or B2 content-list (concatenate text parts)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def rebuild_messages_iankappa(record, mode):
    """Return messages list rebuilt with ian-Kappa prompt style."""
    # Assemble text from existing user message to extract codes
    user_raw = next(
        (m["content"] for m in record["messages"] if m["role"] == "user"), ""
    )
    assembled = _assemble_user_text(user_raw)

    langs   = _LANG_LABEL.findall(assembled)   # ['Python', 'Java']
    codes   = _CODE_BLOCK.findall(assembled)    # [code1, code2]

    if len(langs) < 2 or len(codes) < 2:
        return record["messages"]               # fallback: keep original

    lang1, lang2 = langs[0], langs[1]
    code1, code2 = codes[0], codes[1]
    md1 = lang1.lower().replace("+", "p").replace("#", "sharp").replace(" ", "")
    md2 = lang2.lower().replace("+", "p").replace("#", "sharp").replace(" ", "")
    label = record.get("answer", "Yes")

    header = IANKAPPA_DEFINITION + IANKAPPA_INSTRUCTION

    if mode == "b1":
        user_content = (
            f"{header}\n\n"
            f"Code 1 (language: {lang1}):\n```{md1}\n{code1}\n```\n\n"
            f"Code 2 (language: {lang2}):\n```{md2}\n{code2}\n```\n\nAnswer:"
        )
        return [
            {"role": "system",    "content": IANKAPPA_SYSTEM},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": label},
        ]
    else:  # b2 / b1ctrl / b3
        user_content = [
            {"type": "text",  "text": f"{header}\n\nCode 1 (language: {lang1}):"},
            {"type": "image"},
            {"type": "text",  "text": f"```{md1}\n{code1}\n```\n\nCode 2 (language: {lang2}):"},
            {"type": "image"},
            {"type": "text",  "text": f"```{md2}\n{code2}\n```\n\nAnswer:"},
        ]
        return [
            {"role": "system",    "content": IANKAPPA_SYSTEM},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": label},
        ]


def is_vision_processor(processor):
    """Return True if processor supports image inputs."""
    return hasattr(processor, "image_processor")


def messages_to_text_only(messages):
    """Strip image tokens from multimodal messages for text-only models."""
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [c["text"] for c in content if c.get("type") == "text"]
            result.append({**msg, "content": "".join(text_parts)})
        else:
            result.append(msg)
    return result


def apply_template(processor, messages, add_generation_prompt=True):
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def load_model(checkpoint, base_model, no_lora=False):
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    loaded = False
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
        try:
            import importlib
            cls = getattr(importlib.import_module("transformers"), cls_name)
            base = cls.from_pretrained(base_model, **load_kwargs)
            print(f"Loaded base with {cls_name}")
            loaded = True
            break
        except (ImportError, AttributeError):
            continue
        except Exception as e:
            print(f"{cls_name} failed: {e}")
            continue
    if not loaded:
        raise RuntimeError(f"Could not load {base_model}")

    if no_lora:
        model = base
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        model = PeftModel.from_pretrained(base, checkpoint)
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return model, processor


def parse_answer(decoded_text):
    """Strip thinking block, then check if output starts with Yes/No."""
    text = _THINK_RE.sub("", decoded_text).strip().lower()
    if text.startswith("yes"):
        return 1
    elif text.startswith("no"):
        return 0
    return -1  # unparseable


@torch.no_grad()
def run_inference(model, processor, records, mode, batch_size=4, max_new_tokens=32,
                  prompt_style="default"):
    preds, golds = [], []

    _vision = is_vision_processor(processor)

    for i in tqdm(range(0, len(records), batch_size), desc="Eval"):
        batch = records[i : i + batch_size]
        texts  = []
        images = []

        for rec in batch:
            # Optionally rebuild messages with alternative prompt
            if prompt_style == "iankappa":
                all_msgs = rebuild_messages_iankappa(rec, mode)
            else:
                all_msgs = rec["messages"]
            # Remove assistant turn for generation
            msgs = [m for m in all_msgs if m["role"] != "assistant"]
            # Text-only models: strip image tokens from multimodal messages
            if not _vision:
                msgs = messages_to_text_only(msgs)
            text = apply_template(processor, msgs, add_generation_prompt=True)
            texts.append(text)

            if mode == "b1" or not _vision:
                pass
            elif mode == "b1ctrl":
                images.extend([_BLACK_IMG, _BLACK_IMG])
            else:
                for rel in (rec["image_paths"] or []):
                    images.append(load_image(rel))

        if images:
            enc = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            enc = processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

        pad_id = (
            processor.tokenizer.pad_token_id
            if hasattr(processor, "tokenizer")
            else processor.pad_token_id
        )
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

        new_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)

        for rec, dec in zip(batch, decoded):
            preds.append(parse_answer(dec))
            golds.append(rec["label"])

    return preds, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       required=True, choices=["b1", "b1ctrl", "b2", "b3"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base_model", default="/data1/models/Qwen3.5-4B")
    ap.add_argument("--test_files", nargs="+", required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--no_lora",      action="store_true",
                    help="Use base model only (zero-shot, no LoRA adapter)")
    ap.add_argument("--prompt_style", default="default", choices=["default", "iankappa"],
                    help="Prompt template: default (Yes/No) or iankappa (clone definition)")
    args = ap.parse_args()

    if not args.no_lora and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --no_lora is set")

    records = load_jsonl(args.test_files)
    print(f"Evaluating {len(records):,} examples  mode={args.mode}  no_lora={args.no_lora}")

    model, processor = load_model(args.checkpoint, args.base_model, no_lora=args.no_lora)
    model.eval()

    preds, golds = run_inference(model, processor, records, args.mode, args.batch_size,
                                 prompt_style=args.prompt_style)

    valid_mask = [p != -1 for p in preds]
    p_valid    = [p for p, v in zip(preds, valid_mask) if v]
    g_valid    = [g for g, v in zip(golds, valid_mask) if v]

    metrics = {
        "n_total":       len(records),
        "n_parseable":   len(p_valid),
        "response_rate": len(p_valid) / len(records) if records else 0,
        "f1":            f1_score(g_valid, p_valid, zero_division=0) if p_valid else 0,
        "precision":     precision_score(g_valid, p_valid, zero_division=0) if p_valid else 0,
        "recall":        recall_score(g_valid, p_valid, zero_division=0) if p_valid else 0,
        "accuracy":      accuracy_score(g_valid, p_valid) if p_valid else 0,
        "mode":          args.mode,
        "prompt_style":  args.prompt_style,
        "checkpoint":    args.checkpoint,
        "test_files":    args.test_files,
    }

    print("\nResults:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
