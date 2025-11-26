import math
from .BaseDataLoader import BaseDataLoader
from datasets import load_dataset,Sequence
import datasets
from PIL import Image
import os
from io import BytesIO

from huggingface_hub import snapshot_download

class Jigsaw2D(BaseDataLoader):
    def load_data(self,dataset_path="JXZhou0224/2DJigsaw",
                  batch_size=1,split="test",
                  test_size=-1,
                  view=False,
                  save_dir="./2DJigsaw",
                  image_scaling=1,
                  **kwargs
                ):
        self.save_dir = save_dir
        self.image_scaling = image_scaling
        if os.path.exists(self.save_dir):
            print("Found existing dataset snapshot, skipping download.")
        else:
            print("Downloading dataset snapshot...")
            self.save_dir = snapshot_download(
                repo_id=dataset_path,
                repo_type="dataset",
                local_dir=self.save_dir
            )
        self.data=load_dataset(
                "json",
                data_files=f"{self.save_dir}/{split}.jsonl",
                split="train"
            )
        self.data = self.data.cast_column("images", Sequence(datasets.Image()))
        print(self.data[0])
        self.batch_size = batch_size
        if test_size == -1:
            self.test_size = len(self.data)
        else:
            self.test_size = test_size
            self.data=self.data.select(range(self.test_size))
        self.data = self.data.cast_column("image", datasets.Image())
        self.n = (self.test_size+(batch_size-1)) // batch_size
    def get_samples(self,i):
        st = i * self.batch_size
        ed = min(len(self.data),st+self.batch_size)
        ret = self.data.select(range(st,ed)).to_list()
        for ins in ret:
            img=Image.open(ins["images"][0]["path"])
            img = img.resize((math.floor(img.width / self.image_scaling), math.floor(img.height / self.image_scaling))) # resize proportionally to a small picture
            ins["image"] = img
        return ret