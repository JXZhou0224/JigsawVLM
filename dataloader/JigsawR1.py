from datasets import load_dataset
import datasets
from .BaseDataLoader import BaseDataLoader
from PIL import Image
import base64, io
import random
from qwen_vl_utils import smart_resize
from io import BytesIO
from utils import add_grid, check_match, parse_output, bytes_to_image
from prompts import get_prompt

# from generate_data.process_image import create_jigsaw
import re

random.seed(42)


# assert(check_match([3,0,1,2],2,2,[1,2,3,0]))
def create_jigsaw(image: Image.Image, m: int, n: int, swap_mode="random"):
    assert swap_mode in [
        "random",
        "single",
    ], "swap mode must be one of 'random','single'"
    w, h = smart_resize(height=image.height, width=image.width, factor=2)
    image = image.resize((w, h))
    w, h = image.size
    pw, ph = w // n, h // m

    patches = []
    for i in range(m):
        for j in range(n):
            patch = image.crop((j * pw, i * ph, (j + 1) * pw, (i + 1) * ph))
            patches.append(patch)

    idx = list(range(m * n))
    shuffled = idx[:]
    if swap_mode == "random":
        random.shuffle(shuffled)
    else:  # single
        i, j = random.sample(range(len(shuffled)), 2)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    new_img = Image.new("RGB", (w, h))
    for k, new_k in enumerate(shuffled):
        i, j = divmod(k, n)
        new_i, new_j = divmod(new_k, n)
        new_img.paste(patches[new_k], (j * pw, i * ph))

    swap_log = shuffled
    return new_img, swap_log


viewed = False


def process_data(
    x,
    draw_grid,
    use_prompt,
    use_thinking=False,
    few_shot=False,
    use_tool=False,
    jigsaw_h=2,
    jigsaw_w=2,
    swap_mode="random",
    index_type="1d",
):
    global viewed
    image = x["image"]
    h, w = jigsaw_h, jigsaw_w
    if isinstance(image, list):
        image = image[0]
    image = bytes_to_image(image)
    image, swap_log = create_jigsaw(image, h, w, swap_mode)
    if draw_grid:
        image = add_grid(image, h, w)
        # raise Exception("not implemented draw grid")
    prompt = get_prompt(
        use_prompt,
        use_thinking,
        use_tool,
        few_shot,
        jigsaw_h,
        jigsaw_w,
        index_type,
        not viewed,
    )
    if not viewed:
        viewed = True
    msg = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, {"type": "image"}],
        }
    ]
    # input(image)
    return {"message": msg, "image": image, "swap_log": swap_log, "h": h, "w": w}


class JigsawR1(BaseDataLoader):
    def load_data(
        self,
        dataset_path="jigsaw-r1/coco",
        batch_size=1,
        split="test",
        test_size=-1,
        jigsaw_h=2,
        jigsaw_w=2,
        draw_grid=False,
        use_prompt="default",
        use_thinking=True,
        few_shot=False,
        swap_mode="random",
        index_type="1d",  # or 2d
        use_tool=False,
        view=False,
        **kwargs
    ):
        self.data = load_dataset(dataset_path)[split]
        self.jigsaw_h = jigsaw_h
        self.jigsaw_w = jigsaw_w
        self.draw_grid = draw_grid
        self.batch_size = batch_size
        self.index_type = index_type
        self.use_thinking = use_thinking
        self.use_prompt = use_prompt
        self.use_tool = use_tool
        if test_size == -1:
            self.test_size = len(self.data)
        else:
            self.test_size = test_size
            self.data = self.data.select(range(self.test_size))
        self.data = self.data.map(
            lambda x: process_data(
                x,
                draw_grid,
                use_prompt,
                use_thinking,
                few_shot,
                use_tool,
                jigsaw_h,
                jigsaw_w,
                swap_mode,
                index_type,
            ),
            batched=False,
        )
        self.data = self.data.cast_column("image", datasets.Image())
        self.n = (self.test_size + (batch_size - 1)) // batch_size
        if view:
            self.view()

    def checker(self, case, completion):
        output = parse_output(
            completion, case["h"], case["w"], index_type=self.index_type
        )
        dic = {}
        dic["image_id"] = case["image_id"]
        dic["swap_log"] = case["swap_log"]
        dic["h"] = case["h"]
        dic["w"] = case["w"]
        dic["answer"] = ""
        if output == None:
            dic["format_result"] = False
        else:
            dic["format_result"] = True
            dic["answer"] = output
        result = check_match(output, case["h"], case["w"], case["swap_log"])
        dic["checker_result"] = result
        dic["completion"] = completion
        print("LLM Output:", completion[:200] if len(completion) > 500 else completion)
        print(dic["format_result"], dic["checker_result"])
        return dic

    def get_samples(self, i):
        st = i * self.batch_size
        ed = min(len(self.data), st + self.batch_size)
        ret = self.data.select(range(st, ed)).to_list()
        for ins in ret:
            ins["message"][0]["content"][1]["image"] = Image.open(
                BytesIO(ins["image"]["bytes"])
            )
        return ret

    def view(self):
        for ins in self.data:
            print(ins["swap_log"])
            print(ins["message"][0]["content"][0]["text"])
            print(ins["w"], ins["h"])
            ins["image"].save("view.png", format="PNG")
            command = input("enter to continue,q to quit")
            if command == "q":
                return
