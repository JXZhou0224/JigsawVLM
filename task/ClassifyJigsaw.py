from .BaseTask import BaseTask,TaskFactory
from PIL import Image
import re
import random
random.seed(42)
default="""
You will be given an image, and the image may be divided into patches and randomly permuted, forming a jigsaw. Or the image maybe an intact original image.
Your task is to detect whether the image is an original image or a jigsaw.
"""

thinking_append="""
First, output the thinking process within <think> </think> tags. Then, provide the final
answer within <answer> </answer> tags. The answer should be a single word, either "jigsaw" or "original"
Here is the input image:
"""

no_thinking_append="""
Directly output your answer, a single word, either "jigsaw" or "original"
"""

prompt_list = {
    "thinking":default+thinking_append,
    "no_thinking":default+no_thinking_append
}

def get_prompt(use_prompt):
    if not use_prompt in prompt_list:
        raise ValueError(f"Unknown prompt type {use_prompt}, if you wish to use custom prompt, please specify it in task's prompt_list. current supported types:\n"+str(prompt_list.keys()))    
    else:
        prompt = prompt_list[use_prompt]
    return prompt
    

def parse_output(completion,use_thinking):
    if not use_thinking:
        return completion
    m = re.search(r"<answer>(.*?)</answer>", completion, re.S)
    if not m:
        return None
    text = m.group(1).strip().lower()
    return text
    

@TaskFactory.register("classify_jigsaw")
class ClassifyJigsaw(BaseTask):
    def __init__(self,
                    use_prompt: str = "thinking",
                    use_thinking: bool = True,
                    **kwargs
                ):
        self.use_prompt = use_prompt
        self.use_thinking = use_thinking

    def solve(self, img: Image.Image, ground_truth: str) -> Image.Image:
        """img is 2 by 2 jigsaw.solve by putting jigsaw to the ground truth order"""
        ground_truth = [int(x) for x in ground_truth.strip().split()]
        w, h = img.width // 2, img.height // 2
        new_img = Image.new('RGB', (img.width, img.height))
        for idx, pos in enumerate(ground_truth):
            x_src = (pos % 2) * w
            y_src = (pos // 2) * h
            x_dst = (idx % 2) * w
            y_dst = (idx // 2) * h
            patch = img.crop((x_src, y_src, x_src + w, y_src + h))
            new_img.paste(patch, (x_dst, y_dst))
        return new_img

    def format_input(self, case):
        img: Image.Image = case["image"]
        is_original = random.random() < 0.5
        if case["reward_model"]["ground_truth"] == "0 1\n2 3":
            is_original = True
        elif is_original:
            img=self.solve(img,case["reward_model"]["ground_truth"])
        prompt = get_prompt(
            self.use_prompt,
        )

        message = [
            {"role":"user", "content": [
                {"type":"text",  "text": prompt},
                {"type":"image", "image": img}
            ]}
        ]

        ret = {"message": message, "image": img,"answer": "original" if is_original else "jigsaw"}
        return ret

    def checker(self, case, completion: str):
        guess = parse_output(completion, self.use_thinking)
        ok = False
        if guess is not None:
            print(case)
            ok = (case["answer"] == guess)
        return {
            "ground_truth": case.get("answer"),
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

