import json
import argparse
from tqdm import tqdm
import re
import os
from typing import Literal
from config import DefaultConfig, LocalConfig,CustomConfig
from dataloader import JigsawR1
import pandas as pd
from model import Qwen2_5_VL
from pathlib import Path
import csv

import json
from typing import Iterable, Union, List, Dict

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default="default")
    return parser.parse_args()

def parse_answer(completion, h, w) -> list[int] | None:
    m = re.search(r"<answer>(.*?)</answer>", completion, re.S)
    if not m: return None
    txt = m.group(1).strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if len(lines) != h: return None
    nums: list[int] = []
    try:
        for l in lines:
            parts = l.split()
            if len(parts) != w:
                return None
            nums.extend(int(x) for x in parts)
    except ValueError:
        return None
    if len(nums) != w * h or sorted(nums) != list(range(w * h)):
        return None
    return nums
# print(parse_answer("fdsaf\n<answer>\n\n0 1\n2 3\n</answer>fewaf",2,2))

def append_answer(logs, save_dir):
    save_dir = Path(save_dir)
    df = pd.DataFrame(logs)
    if save_dir.exists() and save_dir.stat().st_size > 0:
        df.to_csv(save_dir, mode='a', header=False, index=False)
    else:
        df.to_csv(save_dir, mode='w', header=True, index=False)

if __name__ == "__main__":
    args = parse_args()
    config = None
    if args.config == "default":
        config=DefaultConfig
    elif args.config == "local":
        config=LocalConfig
    else:
        file:str = args.config
        assert file.endswith(".json"),"config must be a json, or select from existing configs"
        config=CustomConfig.load_from_dict(json.load(file))
    dataloader = None
    if config.dataloader_name == "JigsawR1":
        dataloader = JigsawR1(**config.data_cfg)
    else:
        raise Exception("data not supported")
    
    model = Qwen2_5_VL(**config.model_cfg)
    
    cnt = 0
    save_dir=Path(config.save_dir)
    res_dir = save_dir.joinpath("result.jsonl")
    
    os.makedirs(save_dir,exist_ok=True)
    config.to_json(str(save_dir)+"/config.json")
    if res_dir.exists() and res_dir.stat().st_size > 0:
        processed_len = len(load_jsonl(res_dir))
    else:
        processed_len = 0
    print("starting at case: ",processed_len)
    for sample in tqdm(dataloader): 
        if cnt < processed_len:
            cnt += len(sample)
            continue
        completions = model.generate([ins["message"] for ins in sample])
        logs = dataloader.check(sample,completions)
        dump_jsonl(logs,res_dir,"a")
        cnt += len(sample)
