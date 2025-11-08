from datasets import load_dataset
import datasets
from .BaseDataLoader import BaseDataLoader
from PIL import Image
import base64, io
import random
from qwen_vl_utils import smart_resize
from io import BytesIO
from utils import add_grid
# from generate_data.process_image import create_jigsaw
import re
random.seed(42)

prompts= {
    "rearrange": """The input image is divided into {h}x{w} patches that have been randomly permuted from their original
positions. Your task is to solve this {h}x{w} jigsaw puzzle and reconstruct the original image.
Consider a {h}x{w} grid, where each number represents a position index ranging from {tl} (topleft) to {br} (bottom-right):
{num_label}
For each patch, determine its correct position index in the original image. If a patch currently at position X should belong at position Y, place "Y" at position X.
""",

    "thinking": """First, output the thinking process within <think> </think> tags. Then, provide the final
answer within <answer> </answer> tags. The final answer should be the position indexes arranged
in a {h}x{w} grid.
Here is the input image:""",

    "no_thinking": """Directly output the final answer. The final answer should be the position indexes arranged in a {h}x{w} grid. 
Here is the input image:"""
}

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

def get_num_label(index_type, h, w):
    # top-left and bottom-right "labels" are coordinates or numbers
    if index_type == "1d":
        tl = "0"
        br = str(h * w - 1)
    elif index_type == "2d":
        tl = "(0,0)"
        br = f"({h-1},{w-1})"
    else:
        raise ValueError(f"Unknown index_type {index_type}")

    lines = []
    for i in range(h):
        row = []
        for j in range(w):
            if index_type == "1d":
                row.append(str(i * w + j))
            else:  # 2d
                row.append(f"({i},{j})")
        lines.append(" ".join(row))

    num_label = "\n".join(lines)
    return {"tl": tl, "br": br, "num_label": num_label}

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
def create_jigsaw(image: Image.Image, m: int, n: int):
    w, h = smart_resize(height=image.height,width=image.width,factor=2)
    image = image.resize((w,h))
    w, h = image.size
    pw, ph = w // n, h // m

    patches = []
    for i in range(m):
        for j in range(n):
            patch = image.crop((j * pw, i * ph, (j + 1) * pw, (i + 1) * ph))
            patches.append(patch)

    idx = list(range(m * n))
    shuffled = idx[:]
    random.shuffle(shuffled)

    new_img = Image.new("RGB", (w, h))
    for k, new_k in enumerate(shuffled):
        i, j = divmod(k, n)
        new_i, new_j = divmod(new_k, n)
        new_img.paste(patches[new_k], (j * pw, i * ph))

    swap_log = shuffled
    return new_img, swap_log

def process_data(x,draw_grid,use_prompt,use_thinking=True,index_type="1d"):
    image = x["image"]
    if isinstance(image, list):
        image = image[0]
    image = bytes_to_image(image)
    image,swap_log = create_jigsaw(image,2,2)
    if draw_grid:
        image = add_grid(image,2,2,2)
        # raise Exception("not implemented draw grid")
    h = 2
    w = 2
    prompt = prompts[use_prompt] + prompts["thinking"] if use_thinking else prompts["no_thinking"]
    prompt = prompt.format(h=h,w=w,**get_num_label(index_type,h=h,w=w))
    msg = [
        {"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image"}
        ]}
    ]
    # input(image)
    return {
        "message":msg,
        "image": image,
        "swap_log": swap_log,
        "h": h,
        "w": w
    }

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

class JigsawR1(BaseDataLoader):
    def load_data(self,dataset_path="jigsaw-r1/coco",
                  batch_size=1,split="test",
                  test_size=-1,
                  draw_grid=False,
                  use_prompt="rearrange",
                  use_thinking=True,
                  index_type="1d", # or 2d
                  **kwargs
                ):
        self.data=load_dataset(dataset_path)[split]
        self.draw_grid = draw_grid
        self.batch_size = batch_size
        self.index_type = index_type
        self.use_thinking = use_thinking
        self.use_prompt = use_prompt
        if test_size == -1:
            self.test_size = len(self.data)
        else:
            self.test_size = test_size
            self.data=self.data.select(range(self.test_size))
        self.data = self.data.map(lambda x: process_data(x,draw_grid,use_prompt,use_thinking,index_type),batched=False)
        self.data = self.data.cast_column("image", datasets.Image())
        self.n = (self.test_size+(batch_size-1)) // batch_size
        # self.view()

    def checker(self,case,completion):
        output = parse_output(completion,case["h"],case["w"],index_type=self.index_type)
        dic = {}
        dic["image_id"]=case["image_id"]
        dic["swap_log"]=case["swap_log"]
        dic["h"] = case["h"]
        dic["w"] = case["w"]
        if output == None:
            dic["format_result"] = False
        else:
            dic["format_result"] = True
        result = check_match(output,case["h"],case["w"],case["swap_log"])
        dic["result"]=result
        dic["completion"]=completion
        print(dic["format_result"],dic["result"])
        return dic
    
    def get_samples(self,i):
        st = i * self.batch_size
        ed = min(len(self.data),st+self.batch_size)
        ret = self.data.select(range(st,ed)).to_list()
        for ins in ret:
            ins["message"][0]["content"][1]["image"]=Image.open(BytesIO(ins["image"]["bytes"]))
        return ret

    def view(self):
        for ins in self.data:
            print(ins["swap_log"])
            print(ins["message"][0]["content"][0]["text"])
            print(ins["w"],ins["h"])
            ins["image"].save("view.png",format="PNG")
            input("enter to continue")
            