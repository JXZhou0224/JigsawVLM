from transformers import Qwen2_5_VLForConditionalGeneration,AutoTokenizer, AutoProcessor
from transformers import GenerationConfig, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from .BaseModel import BaseModel
class Qwen2_5_VL(BaseModel):
    def __init__(self,model_name="Qwen/Qwen2.5-VL-3B-Instruct",generation_config={},quantization_config={},**kwargs):
        self.model_name = model_name

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            dtype="auto", 
            device_map="auto",
            generation_config= GenerationConfig(**generation_config),
            quantization_config=BitsAndBytesConfig(**quantization_config),
            **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate(self,messages): # batch of messages or single message
        text = self.processor.apply_chat_template( # this is from ProcessorMixin
            messages, tokenize=False, add_generation_prompt=True
        )
        print(messages)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        return output_text

        

