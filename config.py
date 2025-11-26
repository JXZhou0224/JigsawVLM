# from dataclasses import dataclass,replace
import yaml, json
import copy
def deep_update(base, update):
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

class DefaultConfig():
    save_dir = "result"
    data_cfg = {
        "data_class": "Jigsaw2D",
        "dataset_path": "JXZhou0224/2DJigsaw",
        "split": "test",
        "test_size": 50,
        "batch_size": 1,
        "draw_grid": False,

    }
    task_cfg = {
        "task_class": "solve_jigsaw",
        "jigsaw_h": 2,
        "jigsaw_w": 2,
        "use_prompt": "thinking",
        "use_thinking": True,
        "use_tool": True,
        "index_type": "1d"
    }
    model_cfg = {
        "model_class":"qwen2.5-vl",
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "generation_config":{
                "max_new_tokens":256
            },
        "low_cpu_mem_usage": True
    }

    @classmethod
    def to_json(cls, path):
        data = {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
        data = {}
        for k, v in cls.__dict__.items():
            # skip private attributes, methods and descriptors
            if k.startswith("__"):
                continue
            if isinstance(v, (classmethod, staticmethod)) or callable(v):
                continue
            data[k] = v
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

class LocalConfig(DefaultConfig):
    model_cfg = {
        **DefaultConfig.model_cfg,
        "quantization_config": {
            "load_in_8bit": True,              # enable 8-bit
            "llm_int8_threshold": 6.0,         # optional, controls outlier handling
            "llm_int8_skip_modules": None,     # or list of modules to skip
        }
    }


class CustomConfig(DefaultConfig):
    @classmethod
    def load_from_dict(cls, dic):
        for key, val in dic.items():
            if hasattr(cls, key) and isinstance(getattr(cls, key), dict) and isinstance(val, dict):
                setattr(cls, key, deep_update(getattr(cls, key), val))
            else:
                setattr(cls, key, val)
        return cls