from openai import OpenAI
from .BaseModel import BaseModel, ModelFactory
import base64, io
from PIL import Image

@ModelFactory.register("openai")
class Openai(BaseModel):
    def __init__(self,model_name,temperature=0.0,**kwargs):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def _encode_image(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode('utf-8')

    def _convert_message(self,msg):
        out = []
        for part in msg["content"]:
            if part["type"] == "text":
                out.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                out.append({"type": "image_url", "image_url": {"url": self._encode_image(part["image"])}})
        return {"role": msg["role"], "content": out}

    def generate(self, messages_batch):
        results = []
        for messages in messages_batch:
            formatted = [self._convert_message(m) for m in messages]
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted,
                temperature=self.temperature
            )
            results.append(resp.choices[0].message.content)
        return results