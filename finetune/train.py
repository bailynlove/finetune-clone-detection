#!/usr/bin/env python3
"""
train.py — SFT fine-tuning for B1 / B1-ctrl / B2 / B3 (all use unsloth/Qwen3.5-4B).

Per finetune_plan_v2.md §3.3 (same-model-different-input):
  B1      text-only          (no image token in prompt)
  B1-ctrl black image + text (synthetic 1440×896 black images)
  B2      real image + text  (main method)
  B3      real image-only    (no code text)

Usage:
  CUDA_VISIBLE_DEVICES=2 python finetune/train.py \\
      --mode b1 --model unsloth/Qwen3.5-4B \\
      --train_files dataset/finetune_data/b1/train_a/python_java.jsonl \\
      --val_files   dataset/finetune_data/b1/val/python_java.jsonl \\
      --output_dir  runs/b1_train_a_seed42 --seed 42

LoRA (§5.4): rank=16, alpha=32, dropout=0.05, visual encoder frozen.
Thinking mode disabled via enable_thinking=False in chat template.
"""

import argparse
import json
import os
import random
import re
from io import BytesIO
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

DATASET_DIR = Path("/data1/clone-test/dataset")

# Black image sent for B1-ctrl (same canvas size as rendered images)
_BLACK_IMG = Image.new("RGB", (1440, 896), (0, 0, 0))

# ── Image augmentation (P0 / P1) ──────────────────────────────────────────────

# Dark colour schemes available via Pygments built-in names; bg hex needed for canvas fill.
_AUG_STYLES = [
    ("monokai",     "#272822"),
    ("dracula",     "#282a36"),
    ("github-dark", "#0d1117"),
    ("vim",         "#000000"),
    ("native",      "#202020"),
]
_AUG_FONTS = ["DejaVu Sans Mono", "Liberation Mono"]
_AUG_PADS  = [5, 10, 15, 20]           # image_pad options
_AUG_LPS   = [2, 3, 4, 5]             # line_pad options
_AUG_FONT_TIERS = [(15, 26), (30, 20), (45, 16), (999, 13)]

# Regex to pull code blocks from B2 text parts
_CB_RE   = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
_LH_RE   = re.compile(r"Code \d+ \(language: ([^)]+)\)")
_LANG2LEX = {"Java": "java", "Python": "python", "C++": "cpp", "C#": "csharp",
             "Rust": "rust", "Go": "go", "JavaScript": "javascript"}
_MONOKAI_BG = (39, 40, 34)  # RGB of #272822


def _extract_b2_codes(msgs):
    """Return [(code_str, lexer_name), ...] from a B2 multimodal message list, or None."""
    user = next((m for m in msgs if m["role"] == "user"), None)
    if user is None:
        return None
    content = user["content"]
    text = content if isinstance(content, str) else "".join(
        p.get("text", "") for p in content if p.get("type") == "text"
    )
    langs  = _LH_RE.findall(text)
    blocks = _CB_RE.findall(text)
    if len(langs) < 2 or len(blocks) < 2:
        return None
    return [(blocks[i][1], _LANG2LEX.get(langs[i], "text")) for i in range(2)]


def _render_aug(code, lexer_name, style_name, font_name, image_pad, line_pad):
    """Render code with given Pygments style; returns a 1440×896 RGB PIL Image."""
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import get_lexer_by_name, TextLexer

    TARGET_W, TARGET_H = 1440, 896
    code = re.sub(r"[^\x00-\x7F]", " ", code)
    lines = code.splitlines() or [""]
    n_lines   = len(lines)
    max_chars = max(len(l.expandtabs(4)) for l in lines)

    font_size = _AUG_FONT_TIERS[-1][1]
    for lim, pt in _AUG_FONT_TIERS:
        if n_lines <= lim:
            font_size = pt
            break
    if max_chars > 100:
        pts = [pt for _, pt in _AUG_FONT_TIERS]
        idx = pts.index(font_size)
        font_size = pts[min(idx + 1, len(pts) - 1)]

    try:
        lexer = get_lexer_by_name(lexer_name, stripall=True)
    except Exception:
        lexer = TextLexer()

    bg_hex = dict(_AUG_STYLES).get(style_name, "#272822")
    fmt = ImageFormatter(
        font_name=font_name, font_size=font_size,
        style=style_name, line_numbers=False,
        image_pad=image_pad, line_pad=line_pad,
    )
    try:
        rendered = Image.open(BytesIO(highlight(code, lexer, fmt))).convert("RGB")
    except Exception:
        return Image.new("RGB", (TARGET_W, TARGET_H), bg_hex)

    if rendered.width > TARGET_W or rendered.height > TARGET_H:
        rendered.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), bg_hex)
    canvas.paste(rendered, (0, 0))
    return canvas


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl_files(paths):
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def load_image(rel_path):
    if rel_path == "__black__":
        return _BLACK_IMG.copy()
    full = DATASET_DIR / rel_path
    try:
        return Image.open(full).convert("RGB")
    except Exception:
        return Image.new("RGB", (1440, 896), (39, 40, 34))


# ── Chat template helper ───────────────────────────────────────────────────────

def apply_template(processor, messages, add_generation_prompt=False):
    """Apply chat template; disable thinking mode when the template supports it."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


# ── Collator ──────────────────────────────────────────────────────────────────

class VLMCollator:
    """
    Handles all four modes using the same VLM processor.
    Loss is masked to the final Yes/No answer token only.
    """

    def __init__(self, processor, mode, max_length=4096, max_pixels=None,
                 image_dropout=0.0, image_augment=False):
        self.processor  = processor
        self.mode       = mode
        self.max_length = max_length
        # Limit image resolution to reduce ViT patch count.
        # Qwen2VLImageProcessor uses size['longest_edge'] as the max_pixels cap.
        self.image_dropout = image_dropout
        self.image_augment = image_augment
        if max_pixels is not None and hasattr(processor, "image_processor"):
            ip = processor.image_processor
            if hasattr(ip, "size") and ip.size is not None and "longest_edge" in ip.size:
                ip.size["longest_edge"] = max_pixels
                print(f"Set image_processor.size['longest_edge'] = {max_pixels}")
            elif hasattr(ip, "max_pixels"):
                ip.max_pixels = max_pixels
                print(f"Set image_processor.max_pixels = {max_pixels}")

    def _get_ans_tokenizer(self):
        if hasattr(self.processor, "tokenizer"):
            return self.processor.tokenizer
        return self.processor

    def __call__(self, batch):
        import time
        _t = [time.time()]

        texts      = []
        all_images = []

        for rec in batch:
            msgs = json.loads(rec["messages"]) if isinstance(rec["messages"], str) else rec["messages"]
            text = apply_template(self.processor, msgs, add_generation_prompt=False)
            texts.append(text)

            img_paths = rec["image_paths"]
            if isinstance(img_paths, str):
                img_paths = json.loads(img_paths)

            if self.mode == "b1":
                pass  # no images for text-only
            elif self.mode == "b1ctrl":
                all_images.extend([_BLACK_IMG, _BLACK_IMG])
            else:
                # b2 / b3 — load real images; apply P1 dropout or P0 augmentation
                n_imgs = len(img_paths or []) or 2
                if self.image_dropout > 0 and random.random() < self.image_dropout:
                    # P1: replace entire pair with Monokai background (no signal)
                    blank = Image.new("RGB", (1440, 896), _MONOKAI_BG)
                    all_images.extend([blank] * n_imgs)
                elif self.image_augment:
                    # P0: re-render from code text with random style
                    codes = _extract_b2_codes(msgs)
                    if codes and len(codes) >= 2:
                        style_name, _ = random.choice(_AUG_STYLES)
                        font_name     = random.choice(_AUG_FONTS)
                        image_pad     = random.choice(_AUG_PADS)
                        line_pad      = random.choice(_AUG_LPS)
                        for code, lexer in codes[:2]:
                            all_images.append(_render_aug(
                                code, lexer, style_name, font_name, image_pad, line_pad))
                    else:
                        for rel in (img_paths or []):
                            all_images.append(load_image(rel))
                else:
                    for rel in (img_paths or []):
                        all_images.append(load_image(rel))

        _t.append(time.time())  # after template+images

        if all_images:
            enc = self.processor(
                text=texts,
                images=all_images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            enc = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        _t.append(time.time())  # after processor call
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        if self._call_count <= 5:
            print(f"[COLLATOR timing call={self._call_count}] "
                  f"template+imgs={_t[1]-_t[0]:.2f}s  processor={_t[2]-_t[1]:.2f}s  "
                  f"seq_len={enc['input_ids'].shape[1]}  n_imgs={len(all_images)}", flush=True)

        # Mask loss: keep only the final Yes/No token
        tok = self._get_ans_tokenizer()
        labels = enc["input_ids"].clone()
        for i, rec in enumerate(batch):
            ans_ids = tok.encode(rec["answer"], add_special_tokens=False)
            ids = labels[i].tolist()
            for j in range(len(ids) - len(ans_ids), -1, -1):
                if ids[j : j + len(ans_ids)] == ans_ids:
                    labels[i, :j]                   = -100
                    labels[i, j + len(ans_ids):]    = -100
                    break
            else:
                labels[i] = -100

        enc["labels"] = labels
        return enc


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id, use_4bit=True):
    """Load Qwen3.5-4B (VLM) for all modes."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    # transformers 5.x renamed AutoModelForVision2Seq → AutoModelForImageTextToText
    load_kwargs = dict(
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Use flash_attention_2 to eliminate per-image SDPA Python loop in ViT and
    # speed up LLM full-attention layers. Falls back to eager if unavailable.
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using attn_implementation=flash_attention_2")
    except ImportError:
        print("flash_attn not available, using default attention")

    loaded = False
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
        try:
            import importlib
            cls = getattr(importlib.import_module("transformers"), cls_name)
            model = cls.from_pretrained(model_id, **load_kwargs)
            print(f"Loaded with {cls_name}")
            loaded = True
            break
        except (ImportError, AttributeError):
            continue
        except Exception as e:
            print(f"{cls_name} failed: {e}")
            continue
    if not loaded:
        raise RuntimeError(f"Could not load model {model_id} with any known auto class")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def apply_lora(model, target_modules):
    """Freeze visual encoder / mm_projector; apply LoRA to LLM layers."""
    for name, param in model.named_parameters():
        if any(k in name for k in ("visual", "vision_model", "patch_embed",
                                    "visual_encoder", "mm_projector", "img_projector")):
            param.requires_grad = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def discover_target_modules(model, requested):
    """Return only those module names that actually exist in the model."""
    existing = {n.split(".")[-1] for n, _ in model.named_modules()}
    found = [m for m in requested if m in existing]
    if not found:
        raise ValueError(
            f"None of the requested LoRA target modules {requested} found in model. "
            f"Available leaf names (sample): {list(existing)[:30]}"
        )
    print(f"LoRA target modules: {found}")
    return found


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",           required=True,
                    choices=["b1", "b1ctrl", "b2", "b3"])
    ap.add_argument("--model",          default="unsloth/Qwen3.5-4B")
    ap.add_argument("--train_files",    nargs="+", required=True)
    ap.add_argument("--val_files",      nargs="+", default=None)
    ap.add_argument("--output_dir",     required=True)
    ap.add_argument("--epochs",         type=int,   default=3)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--batch_size",     type=int,   default=4)
    ap.add_argument("--grad_accum",     type=int,   default=4)
    ap.add_argument("--max_length",     type=int,   default=4096)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--no_4bit",        action="store_true")
    ap.add_argument("--max_pixels",     type=int,   default=None,
                    help="Limit image pixels (e.g. 401408 = 784x512) to reduce ViT cost")
    ap.add_argument("--target_modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--limit",          type=int,   default=0,
                    help="If >0, truncate training data to first N records (ablations)")
    ap.add_argument("--image_dropout",  type=float, default=0.0,
                    help="P1: probability to replace a pair's images with blank (0=off)")
    ap.add_argument("--image_augment",  action="store_true",
                    help="P0: re-render images on-the-fly with random style/font/padding")
    return ap.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Mode: {args.mode}  Model: {args.model}  Seed: {args.seed}")

    train_records = load_jsonl_files(args.train_files)
    if args.limit > 0:
        train_records = train_records[:args.limit]
    val_records   = load_jsonl_files(args.val_files) if args.val_files else None
    print(
        f"Train: {len(train_records):,}  Val: {len(val_records):,}"
        if val_records else f"Train: {len(train_records):,}"
    )

    # Serialize complex fields to JSON strings so PyArrow can handle mixed content types
    # (B1ctrl/B2/B3 have list-type user content; B1 has string content — can't mix in Arrow)
    def serialize(rec):
        return {**rec, "messages": json.dumps(rec["messages"]),
                "image_paths": json.dumps(rec["image_paths"])}

    train_ds = Dataset.from_list([serialize(r) for r in train_records])
    val_ds   = Dataset.from_list([serialize(r) for r in val_records]) if val_records else None

    model, processor = load_model(args.model, use_4bit=not args.no_4bit)

    target_modules = discover_target_modules(model, args.target_modules)
    model = apply_lora(model, target_modules)
    model.config.use_cache = False

    # After PEFT wrapping, PEFT's enable_input_require_grads() hooks the LLM inputs to
    # require grad, which pulls the frozen ViT output into the computation graph.
    # HF's gradient_checkpointing_enable() then re-runs every ViT block during backward
    # even though no ViT parameter needs a gradient — doubling ViT compute cost.
    # Fix: disable gradient checkpointing on the visual encoder entirely, and wrap its
    # forward in torch.no_grad() so the output is detached from the graph.
    _base_model = model.base_model.model if hasattr(model, "base_model") else model
    _vis = getattr(getattr(_base_model, "model", _base_model), "visual", None)
    if _vis is not None:
        for m in _vis.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
        _orig_vis_fwd = _vis.forward
        def _no_grad_vis_fwd(*a, **kw):
            with torch.no_grad():
                return _orig_vis_fwd(*a, **kw)
        _vis.forward = _no_grad_vis_fwd
        print(f"ViT GC disabled + wrapped in no_grad ({type(_vis).__name__})")

    collator = VLMCollator(processor, args.mode, max_length=args.max_length,
                          max_pixels=args.max_pixels,
                          image_dropout=args.image_dropout,
                          image_augment=args.image_augment)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=bool(val_ds),
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_length,
        dataset_text_field=None,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
