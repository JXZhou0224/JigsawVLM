from PIL import Image, ImageDraw
import numpy as np
import re

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

def add_grid(img: Image.Image,h_div:int,w_div:int, p: int=4, exclude_margin: bool = False, color=(0, 0, 0)):
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



def check_match(output:list[int],h,w,swap_log:list[int]) -> bool:
    if output == None:
        return False
    
    res = [None]*(w*h)
    print("swap_log:",swap_log)
    for loc,i in enumerate(swap_log):
        res[loc] = output[i]
    print("output:",res)
    if res == list(range(w*h)):
        return True
    return False
# assert(check_match([3,0,1,2],2,2,[1,2,3,0]))
def parse_output(completion, h, w, index_type="1d", use_thinking=True):
    """
    completion: LLM output string
    index_type: "1d" or "2d"
    use_thinking: if True, extract inside <answer>...</answer>, else parse directly
    h, w: needed for flattening 2d -> 1d
    Returns: list[int] or None if format error
    """
    text = completion
    if use_thinking:
        m = re.search(r"<answer>(.*?)</answer>", completion, re.S)
        if not m:
            return None
        text = m.group(1).strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if index_type == "1d":
        try:
            nums = [int(x) for l in lines for x in l.split()]
        except:
            return None
    elif index_type == "2d":
        if h is None or w is None:
            raise ValueError("h and w must be provided for 2d -> 1d")
        try:
            nums_2d = [[eval(x) for x in l.split()] for l in lines]
        except:
            return None
        # flatten 2d -> 1d: i*w + j
        nums = [i * w + j for row in nums_2d for (i, j) in row]
    else:
        raise ValueError("wrong index_type")

    # check numbers are exactly 0..h*w-1
    if sorted(nums) != list(range(h * w)):
        return None

    return nums
