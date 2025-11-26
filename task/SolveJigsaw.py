# tasks/SolveJigsaw.py
from PIL import Image
from io import BytesIO
from utils import add_grid, check_match, parse_output
import json
import base64

from .BaseTask import TaskFactory,BaseTask

default_prompt =  """The input image is divided into {h}x{w} patches that have been randomly permuted from their original
positions. Your task is to solve this {h}x{w} jigsaw puzzle and reconstruct the original image.
Consider a {h}x{w} grid, where each number represents a position index ranging from {tl} (topleft) to {br} (bottom-right):
{num_label}
For each patch, determine its correct position index in the original image. If a patch currently at position X should belong at position Y, place "Y" at position X.
"""

thinking_append = """First, output the thinking process within <think> </think> tags. Then, provide the final
answer within <answer> </answer> tags. The final answer should be the position indexes arranged
in a {h}x{w} grid.
Here is the input image:"""

no_thinking_append = """Directly output the final answer with no explanation. The final answer should be the position indexes arranged in a {h}x{w} grid. 
Here is the input image:"""

tool_append = """You can use the provided tool 'swap_patches' to swap two patches in the image and record your reasoning steps. After the image is put back together. simply output "Done"
Here is the input image:"""

prompt_list= {
    "thinking": default_prompt+thinking_append,
    "no_thinking":default_prompt+no_thinking_append,
    "tool": default_prompt+tool_append
}

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

def get_prompt(
        use_prompt,
        jigsaw_h,
        jigsaw_w,
        index_type="1d", # or 2d
        preview=False
    ):
    if use_prompt not in prompt_list:
        raise ValueError(f"Unknown prompt type {use_prompt}, if you wish to use custom prompt, please specify it in task's prompt_list. current supported types:\n"+str(prompt_list.keys()))
    else:
        prompt = prompt_list[use_prompt]
    prompt = prompt.format(h=jigsaw_h,w=jigsaw_w,**get_num_label(index_type,jigsaw_h,jigsaw_w))
    if preview:
        print(prompt)
    return prompt

class SwapFunc():
    def __init__(self,img,h,w):
        self.img:Image.Image = img
        self.h = h
        self.w = w
        self.patch_w = img.width // w
        self.patch_h = img.height // h
        self.cur_state = list(range(h*w))
    def gen_swapped_image(self, a: int, b: int) -> Image.Image:
        if a == b:
            return self.img

        # Convert id → (row, col)
        ar, ac = divmod(a, self.w)
        br, bc = divmod(b, self.w)

        # Pixel coords for each patch
        ax0 = ac * self.patch_w
        ay0 = ar * self.patch_h
        bx0 = bc * self.patch_w
        by0 = br * self.patch_h

        # Crop patches
        patch_a = self.img.crop((ax0, ay0, ax0 + self.patch_w, ay0 + self.patch_h))
        patch_b = self.img.crop((bx0, by0, bx0 + self.patch_w, by0 + self.patch_h))

        # Paste swapped
        new_img = self.img.copy()
        new_img.paste(patch_b, (ax0, ay0))
        new_img.paste(patch_a, (bx0, by0))

        # Update internal state
        self.img = new_img
        return new_img
    def get_cur_state(self):
        """output a 2d list representing current state"""
        return "\n".join([" ".join([str(ins) for ins in self.cur_state[i:i + self.w]]) for i in range(0, len(self.cur_state), self.w)])
    def get_swap_func(self):
        def swap_patches(reasoning,a,b):
            self.img = self.gen_swapped_image(a,b) 
            self.cur_state[a],self.cur_state[b] = self.cur_state[b],self.cur_state[a]
            buf = BytesIO()
            self.img.save("temp_swapped.png","PNG")
            self.img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return [
                {
                    "type": "text",
                    "text": reasoning
                },
                {
                    "type": "image",
                    "source": {
                        "media_type": "image/png",
                        "data": b64
                    }
                },
                {
                    "type": "text",
                    "text": self.get_cur_state()
                }
            ]
        return swap_patches

    def get_tool(self):
        return {
            "type": "function",
            "name": "swap_patches",
            "description": "swap the two patches a and b, labelled as described by user, starting at 0. and returns the image after swap, as well as the new labels after swap",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
@TaskFactory.register("solve_jigsaw")
class SolveJigsaw(BaseTask):
    def __init__(self,
                 jigsaw_h: int,
                 jigsaw_w: int,
                 index_type: str = "1d",
                 use_prompt: str = "thinking",
                 use_thinking: bool = True,
                 use_tool: bool = False):
        self.h, self.w = jigsaw_h, jigsaw_w
        self.index_type = index_type
        self.use_prompt = use_prompt
        self.use_tool = use_tool
        self.use_thinking = use_thinking
        self.viewed = False


    def format_input(self, case):
        img: Image.Image = case["image"]

        prompt = get_prompt(
            self.use_prompt,
            self.h,
            self.w,
            self.index_type,
            not self.viewed
        )
        self.viewed = True

        message = [
            {"role":"user", "content": [
                {"type":"text",  "text": prompt},
                {"type":"image", "image": img}
            ]}
        ]

        ret = {"message": message, "image": img,"answer": case["reward_model"]["ground_truth"]}
        if self.use_tool:
            swap_func = SwapFunc(img,self.h,self.w)
            tool_func = swap_func.get_swap_func()
            tool = swap_func.get_tool()
            ret["tools"] = [tool]
            ret["tool_map"] = {"swap_patches": tool_func}
        return ret

    def checker(self, case, completion: str):
        guess = parse_output(completion, self.h, self.w, self.index_type,self.use_thinking)
        guess = " ".join([str(guess) for guess in guess]) if guess is not None else None
        ok = False
        if guess is not None:
            print(case)
            ground_truth:str = case["answer"]
            ground_truth = " ".join(ground_truth.split())
            ok = (ground_truth == guess)
        return {
            "image_id": case.get("image_id"),
            "ground_truth": case.get("answer"),
            "h": self.h,
            "w": self.w,
            "guess": guess or "",
            "format_ok": guess is not None,
            "correct": ok,
            "completion": completion
        }

    def view_case(self, case):
        """Optional console‐based viewer."""
        print("Swap log:", case["swap_log"])
        case["image"].show()
        input("ENTER to continue, q to quit: ")

