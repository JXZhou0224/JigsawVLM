from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from JigsawVLM.utils.jigsaw_utils import generate_bw_grid
import torch
def qwen_pipeline():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    '''
    This is Qwen2_5_VLProcessor
    '''
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template( # this is from ProcessorMixin
        messages, tokenize=False, add_generation_prompt=True
    )
    '''
    text has the follow segment:
    <|vision_start|><|image_pad|><|vision_end|>

    <|image_pad|> has this special token, will be replaced by image tokens
    '''

    image_inputs, video_inputs = process_vision_info(messages)
    '''
    image inputs: list of PIL image
    **image resize happens here.** by calling smart resize
    '''
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    TOKEN_PLACEHOLDER_ID=151655
    '''
    input: BatchFeature
    - input_ids: list of token ids
    - attention mask
    - pixel_values: (length,3*14*14*2) 2 stands for 2 temporal frame

    dividing into batch happens here.

    '''
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


def validate_patching():
    img = generate_bw_grid(14,14,4,4)
    # minial size of the graph is 4*4 which gives 2*2 = 4 tokens
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                }
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    text = processor.apply_chat_template( # this is from ProcessorMixin
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    print(img.size)
    print(image_inputs[0].size)
    assert img.width == image_inputs[0].width and img.height == image_inputs[0].height, "resize occurred"
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # torch.set_printoptions(edgeitems=10)
    print(inputs)
    print(set(inputs["pixel_values"][0]))
    print(inputs["pixel_values"][1])


if __name__ == "__main__":
    validate_patching()