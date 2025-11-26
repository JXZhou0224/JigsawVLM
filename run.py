from config import DefaultConfig, LocalConfig,CustomConfig
from dataloader import Jigsaw2D
from task import BaseTask, TaskFactory
from model import ModelFactory,BaseModel
from utils import load_jsonl,dump_jsonl

import json
import argparse
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default="default")
    parser.add_argument('--overwrite',action="store_true",default=False)
    parser.add_argument('--continue_run',action="store_true",default=False)
    return parser.parse_args()

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
        config=CustomConfig.load_from_dict(json.load(open(file,"r")))
    dataloader = None
    if config.data_cfg["data_class"] == "Jigsaw2D":
        dataloader = Jigsaw2D(**config.data_cfg)
    else:
        raise Exception("data not supported")
    
    task = TaskFactory.get_task(**config.task_cfg)

    model = ModelFactory.get_llm(**config.model_cfg)
    if args.overwrite and args.continue_run:
        raise Exception("only one of the --overwrite and --continue_run should be used")
    cnt = 0
    save_dir=Path(config.save_dir)
    res_dir = save_dir.joinpath("result.jsonl")
    os.makedirs(save_dir,exist_ok=True)
    if res_dir.exists() and res_dir.stat().st_size > 0:
        if args.overwrite:
            processed_len = 0
            os.remove(res_dir)
        elif args.continue_run:
            processed_len = len(load_jsonl(res_dir))
        else:
            raise Exception(f"save_dir {save_dir} not empty, please specify you want to overwrite or continue")
    else:
        processed_len = 0    
    config.to_json(str(save_dir)+"/config.json")

    print("starting at case: ",processed_len)
    for samples in tqdm(dataloader): 
        if cnt < processed_len:
            cnt += len(samples)
            continue
        inputs = task.format_inputs(samples)
        completions = model.generate(inputs)
        logs = task.check(inputs,completions)
        dump_jsonl(logs,res_dir,"a")
        cnt += len(samples)
