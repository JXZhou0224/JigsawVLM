from PIL import Image
from typing import Literal, Callable
import random
import math
import json
import os
from qwen_vl_utils import smart_resize

def choose_adjacent_patch(index: int, m: int, n: int, exclude: set[int] = None) -> int:
    """Choose an adjacent patch index for a given patch index in an m x n grid.
    The function tries to avoid indices in the exclude list if provided.
    """
    assert 0 <= index < m * n, "Index out of bounds"
    row, col = divmod(index, n)
    adjacent_indices = []
    # Up
    if row > 0:
        adjacent_indices.append((row - 1) * n + col)
    # Down
    if row < m - 1:
        adjacent_indices.append((row + 1) * n + col)
    # Left
    if col > 0:
        adjacent_indices.append(row * n + (col - 1))
    # Right
    if col < n - 1:
        adjacent_indices.append(row * n + (col + 1))
    excluded_adjacent_indices = []
    if exclude:
        excluded_adjacent_indices = [i for i in adjacent_indices if i not in exclude]
    if len(excluded_adjacent_indices) > 0:
        # If there are adjacent patches not in exclude, choose from them
        return random.choice(excluded_adjacent_indices)
    # Otherwise, choose from all adjacent patches
    return random.choice(adjacent_indices) if adjacent_indices else index   

def choose_any_patch(index: int, m: int, n: int, exclude: set[int] = None) -> int:
    """Choose any patch index different from the given index."""
    choices = list(range(m * n))
    choices.remove(index)
    return random.choice(choices)

def swap_patches(
    img:Image.Image,
    m: int,
    n: int,
    num_of_swap: int,
    swap_mode: Literal["random", "clustered"] = "random",
    seed: int = None
):
    """
    Split image into m x n patches, perform k swaps, and save result.
    Args:
        img (Image.Image): Path to input image.
        m, n (int): Grid division for rows and columns.
        num_of_swap (int): Number of patch swaps.
        swap_mode (str): "random" or "clustered".
        seed (int): Random seed (optional).
    """
    if seed is not None:
        random.seed(seed)

    assert swap_mode in ["random", "clustered"], "swap_area must be 'random' or 'clustered'"
    select_patch_fn: Callable[[int, int, int], int] = choose_adjacent_patch if swap_mode == "clustered" else choose_any_patch

    w, h = img.size
    assert w % n == 0 and h % m == 0, "m and n must be a factor of image's width and height. Please resize image before calling"
    patch_w, patch_h = w // n, h // m

    patches = []            

    # --- 1. Split image into patches ---
    patches = []
    for i in range(m):
        for j in range(n):
            box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            patches.append(img.crop(box))

    total_patches = len(patches)
    swap_log = []

    # --- 2. Perform swaps ---
    if swap_mode == "random":
        for _ in range(num_of_swap):
            i = random.randrange(total_patches)
            j = select_patch_fn(i, m, n)
            patches[i], patches[j] = patches[j], patches[i]
            swap_log.append((i, j))

    elif swap_mode == "clustered":
        # choose an initial patch
        anchor = random.randrange(total_patches)
        previous_swapped = {anchor}
        for _ in range(num_of_swap):
            j = select_patch_fn(anchor, m, n, exclude=previous_swapped)
            patches[anchor], patches[j] = patches[j], patches[anchor]
            swap_log.append((anchor, j))
            previous_swapped.add(j)
            anchor = random.choice(list(previous_swapped))

    # --- 3. Reconstruct swapped image ---
    swapped_img = Image.new("RGB", (w, h))
    for idx, patch in enumerate(patches):
        i, j = divmod(idx, n)
        swapped_img.paste(patch, (j * patch_w, i * patch_h))

    return swapped_img, swap_log

PATCH_SIZE = 14
FACTOR = 2 * PATCH_SIZE
def calculate_resize(w,h,h_div,factor=FACTOR) -> tuple[int,int]:
    assert h_div >= 2, "must be divided into at least 2 pieces"
    w_div = w*h_div/h
    assert w_div >= 2, "individable, change to larger h_dive"
    max_pixels = h_div*factor*(factor*math.ceil(w_div))
    return smart_resize(w,h,factor,max_pixels=max_pixels)

def create_jigsaw(img:Image.Image,h_div=4,factor=FACTOR,num_of_swap=1,seed=42) ->tuple[Image.Image, list[tuple[int,int]]]:
    w, h = img.size
    resized_w, resized_h = calculate_resize(w,h,h_div,factor)
    img = img.resize((resized_w, resized_h))
    img,swap_log = swap_patches(img,resized_h//factor,resized_w//factor,num_of_swap,seed=42)
    return img,swap_log

if __name__ == "__main__":
    # Example usage
    img = Image.open("example.png")
    img, swap_log = create_jigsaw(img,4,num_of_swap=1)
    img.save("example_out.png",format="PNG")