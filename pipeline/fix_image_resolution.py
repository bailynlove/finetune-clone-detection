#!/usr/bin/env python3
"""
fix_image_resolution.py
Pad all rendered code images to a fixed 1120×896 canvas (1280 patches @ 28×28).

Before: variable-size images (204-888px wide, 59-862px tall, 1844 unique sizes)
After:  all images 1120×896px, code anchored top-left, remainder filled with
        Monokai background #272822

No cropping: images currently larger than the canvas (rare, due to thumbnail()) are
scaled down with LANCZOS before pasting.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

IMAGES_DIR = Path("/data1/clone-test/dataset/images")
TARGET_W   = 1120
TARGET_H   = 896
BG_COLOR   = (39, 40, 34)   # #272822 Monokai background


def pad_one(img_path):
    img_path = Path(img_path)
    try:
        img = Image.open(img_path).convert("RGB")

        # Scale down only if the rendered image somehow exceeds the canvas
        if img.width > TARGET_W or img.height > TARGET_H:
            img.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)

        # Already exact size — skip
        if img.width == TARGET_W and img.height == TARGET_H:
            return "skip"

        # Create fixed canvas and paste code at top-left
        canvas = Image.new("RGB", (TARGET_W, TARGET_H), BG_COLOR)
        canvas.paste(img, (0, 0))
        canvas.save(img_path, "PNG", optimize=True, compress_level=6)
        return "ok"
    except Exception as e:
        return f"error:{e}"


def main():
    all_imgs = list(IMAGES_DIR.rglob("*.png"))
    print(f"Images to process: {len(all_imgs):,}")

    # Quick before-stats on sample
    import random
    rng = random.Random(42)
    sample = rng.sample(all_imgs, min(500, len(all_imgs)))
    sizes_before = set()
    for p in sample:
        try:
            sizes_before.add(Image.open(p).size)
        except Exception:
            pass
    print(f"Unique sizes before (sample of {len(sample)}): {len(sizes_before)}")

    from collections import Counter
    results = Counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(pad_one, str(p)) for p in all_imgs]
        for fut in tqdm(as_completed(futures), total=len(all_imgs), desc="Padding"):
            base = fut.result().split(":")[0]
            results[base] += 1

    print("\nResults:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:,}")

    # Verify after
    sample2 = rng.sample(all_imgs, min(500, len(all_imgs)))
    sizes_after = set()
    for p in sample2:
        try:
            sizes_after.add(Image.open(p).size)
        except Exception:
            pass
    print(f"\nUnique sizes after (sample of {len(sample2)}): {len(sizes_after)}")
    print(f"All images now {TARGET_W}×{TARGET_H}: {sizes_after == {(TARGET_W, TARGET_H)}}")


if __name__ == "__main__":
    main()
