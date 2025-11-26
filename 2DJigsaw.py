#!/usr/bin/env python3
"""
create_2d_jigsaw.py

Creates a 2DJigsaw dataset from COCO train2014 (sampled 9k: 8k train, 1k val) and a test split from
Hugging Face dataset "jigsaw-r1/coco" (expected ~1k). Produces images (2x2 permuted jigsaws) and a JSONL
with rows of the following format:

{
  "data_source": "2DJigsaw",
  "prompt": [{"role": "user", "content": "<image>\n" + instruction_prompt}],
  "images": ["2DJigsaw/images/<split>/name_of_the_image.png"],
  "ability": "jigsaw",
  "reward_model": {"style": "rule", "ground_truth": <ground_truth>},
  "extra_info": {
     "split": "train_rl",
     "index": 24,
     "answer": <ground_truth>,
     "question": "<image>\n" + instruction_prompt,
     "swap_mode": "1 or 2 or random"
  }
}

Notes / defaults chosen per your request:
- target repo left blank (no automatic HF upload).
- instruction_prompt left blank.
- qwen_vl_utils.smart_resize is called with no params (if qwen_vl_utils is not installed, a fallback smart_resize is used).
- swap modes: "1" = one transposition, "2" = two sequential transpositions, "random" = fully random permutation.
- ground_truth is stored as a JSON list (e.g. [2,0,3,1]) representing the original patch indices in the order they appear
  in the permuted image (top-left -> top-right -> bottom-left -> bottom-right). This is the form we expect an LLM to output.
- The script saves progress incrementally (image files and JSONL lines appended) so you can stop/restart safely.
- A preview function dumps a few temp PNGs and prints the corresponding JSON rows to stdout for inspection.

Usage:
    python create_2d_jigsaw.py --outdir ./2DJigsaw_data --sample-size 9000 --preview-n 5

Requirements (install via pip if missing):
    pip install pillow numpy tqdm requests datasets huggingface_hub

"""
import os
import io
import sys
import json
import random
import argparse
import tempfile
import base64
import shutil
from zipfile import ZipFile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

SEED = 42

instruction_prompt = """
The input image is divided into 2x2 patches that have been randomly permuted from their original
positions. Your task is to solve this 2x2 jigsaw puzzle and reconstruct the original image.
Consider a 2x2 grid, where each number represents a position index ranging from 0 (topleft) to 3 (bottom-right):
0 1
2 3
For each patch, determine its correct position index in the original image. If a patch currently at position X should belong at position Y, place "Y" at position X.
First, output the thinking process within <think> </think> tags. Then, provide the final
answer within <answer> </answer> tags. The final answer should be the position indexes arranged
in a 2x2 grid.
"""

# Try optional imports
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import requests
except Exception:
    requests = None

random.seed(SEED)
np.random.seed(SEED)

# --------- Utility / resize wrapper ---------
def smart_resize_pillow(img: Image.Image, max_size: int = 1536) -> Image.Image:
    """
    Fallback smart_resize:
    - Resizes the image so its longer edge <= max_size while keeping aspect ratio.
    - Ensures resulting width and height are divisible by 2.
    """
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
    else:
        new_w, new_h = w, h

    # make divisible by 2
    if new_w % 2 != 0:
        new_w -= 1
    if new_h % 2 != 0:
        new_h -= 1

    if new_w <= 0:
        new_w = 2
    if new_h <= 0:
        new_h = 2

    return img.resize((new_w, new_h), Image.LANCZOS)


def get_smart_resize():
    """
    Try to import qwen_vl_utils.smart_resize. If available, return it wrapped to ensure even dims.
    Otherwise return fallback smart_resize_pillow with defaults.
    """
    try:
        import qwen_vl_utils as qvu  # type: ignore
        if hasattr(qvu, "smart_resize"):
            def _sr(img: Image.Image):
                h,w = qvu.smart_resize(img.height,img.width,2)  # call with no args per user request (default behavior)
                out = img.resize((w, h), Image.LANCZOS)
                return out
            return _sr
    except Exception:
        pass

    # fallback
    return smart_resize_pillow


smart_resize = get_smart_resize()


# --------- Jigsaw permutation generation ---------
def gen_perm_swap_mode(mode: str) -> List[int]:
    """
    Generate a permutation of [0,1,2,3] according to swap_mode:
      - "1": perform exactly one swap (one transposition) from identity.
      - "2": perform exactly two sequential swaps (may produce identity in rare cases).
      - "random": completely random permutation (could be identity).
    Returns a list of 4 ints representing the original indices in the order they appear in the permuted image.
    Example: [2,0,3,1] means that top-left patch in permuted image comes from original patch 2, etc.
    """
    base = [0, 1, 2, 3]
    if mode == "1":
        a, b = random.sample(range(4), 2)
        perm = base.copy()
        perm[a], perm[b] = perm[b], perm[a]
        return perm
    elif mode == "2":
        perm = base.copy()
        for _ in range(2):
            a, b = random.sample(range(4), 2)
            perm[a], perm[b] = perm[b], perm[a]
        return perm
    elif mode == "random":
        perm = base.copy()
        random.shuffle(perm)
        return perm
    else:
        raise ValueError("Unknown swap mode: " + str(mode))


# --------- Image patching / assembling ---------
def split_2x2(img: Image.Image) -> List[Image.Image]:
    """
    Split image into 2x2 patches and return list of patches in original order [0..3] corresponding to:
        0 | 1
        -----
        2 | 3
    """
    w, h = img.size
    mw, mh = w // 2, h // 2
    patches = [
        img.crop((0, 0, mw, mh)),          # 0 top-left
        img.crop((mw, 0, w, mh)),          # 1 top-right
        img.crop((0, mh, mw, h)),          # 2 bottom-left
        img.crop((mw, mh, w, h)),          # 3 bottom-right
    ]
    return patches


def assemble_from_perm(patches: List[Image.Image], perm: List[int]) -> Image.Image:
    """
    Given original patches [0..3] and perm (list of original indices in order they should appear),
    assemble a new 2x2 image with that order (top-left to bottom-right).
    """
    # assume all patches same size
    pw, ph = patches[0].size
    new = Image.new("RGB", (pw * 2, ph * 2))
    order = perm  # perm gives original indices in the order of positions
    coords = [
        (0, 0),
        (pw, 0),
        (0, ph),
        (pw, ph),
    ]
    for pos_idx, orig_idx in enumerate(order):
        patch = patches[orig_idx]
        new.paste(patch, coords[pos_idx])
    return new


# --------- IO / dataset pipeline ---------
def download_and_sample_coco_train(sample_n: int, data_dir: str) -> List[Tuple[str, bytes]]:
    """
    Modified: DOES NOT download or unzip anything.
    Instead, assumes train2014/ already exists inside tmp_dir.
    We simply list images and sample sample_n of them.
    """
    import random
    random.seed(42)

    train_dir = data_dir  # your extracted folder
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Expected directory: {train_dir}")

    # list all image files
    all_imgs = []
    for fname in os.listdir(train_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_imgs.append(fname)

    if len(all_imgs) == 0:
        raise RuntimeError(f"No images found inside {train_dir}")

    if sample_n > len(all_imgs):
        raise ValueError(f"sample_n = {sample_n} > total images = {len(all_imgs)}")

    # deterministic order → deterministic sampling
    all_imgs = sorted(all_imgs)
    sampled = random.sample(all_imgs, sample_n)

    # return (filename, bytes)
    out = []
    for fname in sampled:
        fpath = os.path.join(train_dir, fname)
        with open(fpath, "rb") as f:
            out.append((fname, f.read()))

    return out

def base64_to_bytes(image_base64):
    return base64.b64decode(image_base64)

def bytes_to_image(element):
    if isinstance(element, bytes):
        image = Image.open(io.BytesIO(element))
        return image
    elif isinstance(element, str):
        image_bytes = base64_to_bytes(element)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    elif isinstance(element, Image.Image):
        return element
    else:
        raise TypeError(f"{type(element)} | {element}")

def save_image_bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def safe_basename(orig_name: str) -> str:
    # remove extension and keep a filesystem-safe name
    base = os.path.splitext(orig_name)[0]
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in base)

def inverse_perm(p):
    inv = [0] * len(p)
    for i, v in enumerate(p):
        inv[v] = i
    return inv


# --------- Main pipeline to create dataset rows and images ---------
def create_2d_jigsaw_dataset(
    outdir: str,
    coco_zip_url: str = "http://images.cocodataset.org/zips/train2014.zip",
    sample_size: int = 9000,
    train_n: int = 8000,
    val_n: int = 1000,
    test_hf_id: str = "jigsaw-r1/coco",
    preview_n: int = 5,
    random_seed: int = 42,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    outdir = Path(outdir)
    images_root = outdir / "images"
    jsonl_root = outdir / "jsonl"
    ensure_dir(images_root)
    ensure_dir(jsonl_root)
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    # Step A: sample train/val from COCO zip
    data_dir = "data/train2014"
    try:
        sampled = download_and_sample_coco_train(sample_size, data_dir)
    except Exception:
        # If download failed or not desired, allow user to provide local extracted images by placing them in tmp_dir/images_source/
        print("Failed to download/extract COCO train zip. Exiting.")
        # shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Split into train/val
    assert sample_size == len(sampled)
    train_items = sampled[:train_n]
    val_items = sampled[train_n:train_n + val_n]

    splits = {
        "train": train_items,
        "val": val_items
    }

    # Step B: test split: load from HF dataset jigsaw-r1/coco if available
    test_items = []
    if load_dataset is not None:
        try:
            print(f"Loading test dataset from Hugging Face: {test_hf_id} ...")
            ds = load_dataset(test_hf_id, split="test")
            # Expect dataset provides image bytes or URLs; we'll try to read "image" column if exists.
            # We'll collect up to ~1000 items
            for i, ex in enumerate(ds):
                if "image" in ex:
                    image = ex["image"]
                    # datasets Image object has .data which might be path-like or PIL
                    try:
                        if isinstance(image, list):
                            image = image[0]
                        image = bytes_to_image(image)
                        buf = io.BytesIO()
                        image.convert("RGB").save(buf, format="PNG")
                        test_items.append((f"hf_test_{i}.png", buf.getvalue()))
                    except Exception:
                        # fallback skip
                        print("error")
                        continue
                # stop when we reach 1000 roughly
                if len(test_items) >= 1000:
                    break
        except Exception as e:
            print("Warning: failed to load test split from HF dataset:", e)
            test_items = []
    else:
        print("datasets library not available; skipping HF test fetch.")

    splits["test"] = test_items

    # Prepare JSONL writers (append mode to support resume)
    jsonl_files = {
        "train": open(jsonl_root / "2DJigsaw_train.jsonl", "a", encoding="utf-8"),
        "val": open(jsonl_root / "2DJigsaw_val.jsonl", "a", encoding="utf-8"),
        "test": open(jsonl_root / "2DJigsaw_test.jsonl", "a", encoding="utf-8"),
    }

    index_counters = {"train": 0, "val": 0, "test": 0}

    # For resume safety, count existing lines to set index counters
    for split in ("train", "val", "test"):
        path = jsonl_root / f"2DJigsaw_{split}.jsonl"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
                index_counters[split] = lines

    # create a small preview collection
    preview_rows = []

    # iterate splits and produce dataset
    for split, items in splits.items():
        if len(items) == 0:
            print(f"Skipping split {split}: no items.")
            continue
        split_img_dir = images_root / split
        ensure_dir(split_img_dir)

        writer = jsonl_files[split]
        for idx_in_split, (orig_name, img_bytes) in enumerate(tqdm(items, desc=f"Processing {split}")):
            # overall index for extra_info
            global_index = index_counters[split]
            index_counters[split] += 1

            try:
                pil = save_image_bytes_to_pil(img_bytes)
            except Exception:
                # skip unreadable images
                continue

            # apply smart_resize (with default params per user's request)
            pil_rs = smart_resize(pil)

            # ensure both dims even
            w, h = pil_rs.size
            if w % 2 != 0 or h % 2 != 0:
                w -= w % 2
                h -= h % 2
                pil_rs = pil_rs.resize((w, h), Image.LANCZOS)

            # split into patches
            patches = split_2x2(pil_rs)

            # choose a swap_mode randomly among "1","2","random" (each image used only once)
            swap_mode = random.choice(["1", "2", "random"])
            perm = gen_perm_swap_mode(swap_mode)

            # assemble new image according to perm
            jigsaw_img = assemble_from_perm(patches, perm)

            # create filename including swap mode and perm string
            base_safe = safe_basename(orig_name)
            permstr = "".join(str(i) for i in perm)
            out_filename = f"{base_safe}_swap{swap_mode}_perm{permstr}.png"
            out_path = split_img_dir / out_filename
            jigsaw_img.save(out_path, format="PNG")

            # ground_truth: JSON list representing original indices in current order (top-left..bottom-right)
            ground_truth = inverse_perm(perm)  # e.g. [2,0,3,1]

            # build JSON row per requirement

            prompt_content = "<image>\n" + instruction_prompt
            row = {
                "data_source": "2DJigsaw",
                "prompt": [{"role": "user", "content": prompt_content}],
                "images": [f"2DJigsaw/images/{split}/{out_filename}"],
                "ability": "jigsaw",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": f"{'train_rl' if split == 'train' else split}",
                    "index": global_index,
                    "answer": ground_truth,
                    "question": prompt_content,
                    "swap_mode": swap_mode
                }
            }

            # append JSONL line immediately (progress saved on the go)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.flush()

            # collect a few preview rows
            if len(preview_rows) < preview_n:
                preview_rows.append((out_path, row))

    # close writers
    for w in jsonl_files.values():
        w.close()

    # produce preview: dump images to temporary PNGs and print JSON rows to stdout
    print("\n=== PREVIEW ===\n")
    for ppath, row in preview_rows:
        # print image path (also copy to tmp so user can inspect easily)
        tmp_preview = Path(tempfile.gettempdir()) / ("2DJigsaw_preview_" + ppath.name)
        shutil.copy(ppath, tmp_preview)
        print("Preview image saved to:", tmp_preview)
        print("JSON row:")
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print("-" * 60)

    print(f"\nDataset creation complete. Output directory: {outdir}")
    print("Images saved under:", images_root)
    print("JSONL files saved under:", jsonl_root)
    print("You can now inspect the preview files printed above. No HF upload was performed by this script.")


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Create 2DJigsaw dataset from COCO train2014 and HF test.")
    p.add_argument("--outdir", type=str, default="./2DJigsaw_data", help="Output directory")
    p.add_argument("--coco-zip-url", type=str, default="http://images.cocodataset.org/zips/train2014.zip", help="COCO train zip URL")
    p.add_argument("--sample-size", type=int, default=9000, help="Total number of images sampled from COCO train (train+val)")
    p.add_argument("--train-n", type=int, default=8000, help="Number of train images from sampled set")
    p.add_argument("--val-n", type=int, default=1000, help="Number of val images from sampled set")
    p.add_argument("--test-hf-id", type=str, default="jigsaw-r1/coco", help="Hugging Face dataset id for test (optional)")
    p.add_argument("--preview-n", type=int, default=5, help="Number of preview examples to print")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_2d_jigsaw_dataset(
        outdir=args.outdir,
        coco_zip_url=args.coco_zip_url,
        sample_size=args.sample_size,
        train_n=args.train_n,
        val_n=args.val_n,
        test_hf_id=args.test_hf_id,
        preview_n=args.preview_n,
        random_seed=args.seed,
    )
