from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import json


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,              # enable 8-bit
    llm_int8_threshold=6.0,         # optional, controls outlier handling
    llm_int8_skip_modules=None,     # or list of modules to skip
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    dtype="auto", 
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


def run(messages):
    text = processor.apply_chat_template( # this is from ProcessorMixin
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print(inputs["input_ids"][0].tolist().count(151655))
    print(inputs["image_grid_thw"])
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

while True:
    inp = input("enter to run, q to quit")
    if inp == "q":
        exit(0)
    run(json.load(open("msg.json")))