from PIL import Image
import base64, io
import json
from typing import List, Dict,Iterable,Union
def load_jsonl(path: str) -> List[Dict]:
    """
    Read a .jsonl file and return a list of dicts.
    """
    objs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    return objs

def dump_jsonl(
    data: Iterable[Union[Dict, List]],
    path: str,
    mode: str = "a"
) -> None:
    """
    Write an iterable of JSON-serializable objects to a .jsonl file.
    mode: "w" to overwrite, "a" to append.
    """
    with open(path, mode, encoding='utf-8') as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

def image_to_base64(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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