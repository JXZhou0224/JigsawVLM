from openai import OpenAI
from .BaseModel import BaseModel, ModelFactory
import base64, io
from PIL import Image
import json

@ModelFactory.register("openai")
class Openai(BaseModel):
    def __init__(self,model_name,temperature=0.0,**kwargs):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def _encode_image(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(open("temp.png","wb"))
        img.save(buf, format="PNG")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode('utf-8')

    def _convert_message(self,msg):
        out = []
        for part in msg["content"]:
            if part["type"] == "text":
                out.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image":
                out.append({"type": "input_image", "image_url": self._encode_image(part["image"])})
        return {"role": msg["role"], "content": out}

    def generate(self, inputs_batch):
        results = []
        for case in inputs_batch:
            if("tools" not in case):
                formatted = [self._convert_message(m) for m in case["message"]]
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=formatted,
                    temperature=self.temperature
                )
                results.append(resp.output[0].content[0].text)
                print("Response:", resp.output[0].content[0].text)
            else:
                history = [self._convert_message(m) for m in case["message"]]
                while True:
                    log = []
                    resp = self.client.responses.create(
                        model=self.model_name,
                        input=history,
                        tools=case["tools"]
                    )
                    history+=resp.output
                    for item in resp.output:
                        if item.type == "function_call":
                            print("Function call:", item.name, item.arguments)
                            output = case["tool_map"][item.name](**json.loads(item.arguments))
                            log.append(item.arguments)
                            history.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps({
                                item.name: output
                                })
                            })
                    if all(item.type != "function_call" for item in resp.output):
                        res = "\n".join(log)
                        results.append(res)
                        break
        return results