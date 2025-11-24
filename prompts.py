default_prompt = """The input image is divided into {h}x{w} patches that have been randomly permuted from their original
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

few_shot_example_1d_2x2 = """Example: For a 2x2 jigsaw puzzle with positions labeled 0-3:
0 1
2 3

If the patches are currently arranged as:
3 0
1 2

Then the correct position mapping is:
1 2
3 0

This means: patch at position 0 belongs at position 1, patch at position 1 belongs at position 2, etc.
"""

few_shot_example_1d_3x3 = """Example: For a 3x3 jigsaw puzzle with positions labeled 0-8:
0 1 2
3 4 5
6 7 8

If the patches are currently arranged as:
5 3 7
8 0 1
2 6 4

Then the correct position mapping is:
5 3 7
8 0 1
2 6 4

This means the patch at each current position should be moved to show where it belongs in the original image.
"""

few_shot_prompt_suffix = """{example}
Now solve the following jigsaw puzzle. Remember: for each patch at position X, output where it should go in the original image arrangement."""

prompt_list = {
    "default": default_prompt,
    "thinking": thinking_append,
    "no_thinking": no_thinking_append,
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
    use_prompt="default",
    use_thinking=True,
    use_tool=False,
    few_shot=False,
    jigsaw_h=2,
    jigsaw_w=2,
    index_type="1d",  # or 2d
    preview=False,
):
    if use_tool:
        raise Exception("not implemented")
    if use_prompt != "default":
        prompt = use_prompt
        print("\033[94mUsing the following customized prompt:\033[0m")
        print(prompt)
        print("*" * 10)
    else:
        prompt = prompt_list["default"]

    if few_shot:
        if index_type == "1d":
            if jigsaw_h == 2 and jigsaw_w == 2:
                example = few_shot_example_1d_2x2
            elif jigsaw_h == 3 and jigsaw_w == 3:
                example = few_shot_example_1d_3x3
            else:
                example = few_shot_example_1d_2x2
        else:
            example = (
                few_shot_example_1d_2x2
                + "\n(Note: In 2d format, positions are represented as (row, col) coordinates.)"
            )

        prompt += "\n" + few_shot_prompt_suffix.format(example=example)

    prompt += prompt_list["thinking"] if use_thinking else prompt_list["no_thinking"]
    prompt = prompt.format(
        h=jigsaw_h, w=jigsaw_w, **get_num_label(index_type, jigsaw_h, jigsaw_w)
    )
    if preview:
        print(prompt)
    return prompt
