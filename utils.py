from PIL import Image, ImageDraw
import numpy as np

def generate_bw_grid(m=14, n=14, M=4, N=4,save_path=""):
    """
    Generate a black-white alternating RGB grid image.
    
    Args:
        m, n: block size in pixels (height, width)
        M, N: number of block repetitions vertically and horizontally
        save_path: where to save the image (jpg)
    Returns:
        img: a PIL Image object
    """
    # Each block is either black [0,0,0] or white [255,255,255]
    block = np.zeros((M*m, N*n, 3), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            color = 255 if (i + j) % 2 == 0 else 0
            block[i*m:(i+1)*m, j*n:(j+1)*n] = color
    img = Image.fromarray(block, mode="RGB")
    if save_path != "":
        img.save(save_path, "PNG")
    return img

def add_grid(img: Image.Image,h_div:int,w_div:int, p: int, exclude_margin: bool = False, color=(0, 0, 0)):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    assert h % h_div == 0 and w % w_div == 0
    inc_h = h // h_div
    inc_w = w // w_div
    for i in range(1, w_div):
        x = i * inc_w
        draw.rectangle([(x - p, 0), (x + p, h)], fill=color)
    for j in range(1, h_div):
        y = j * inc_h
        draw.rectangle([(0, y - p), (w, y + p)], fill=color)

    if not exclude_margin:
        draw.rectangle([(0, 0), (w, p)], fill=color)
        draw.rectangle([(0, h - p), (w, h)], fill=color)
        draw.rectangle([(0, 0), (p, h)], fill=color)
        draw.rectangle([(w - p, 0), (w, h)], fill=color)
    return img


def generate_num_label(h,w):
    return "\n".join(" ".join(str(i * w + j) for j in range(w)) for i in range(h))
