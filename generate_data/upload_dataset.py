from datasets import Dataset, DatasetDict
import pandas as pd
from datasets import Image

df = pd.read_csv("patch_jigsaw.csv")
image_dir = "patch_jigsaw/"
dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda x: {"image": image_dir+x["image"]})
dataset = dataset.cast_column("image", Image())

# split into train (80%), val (10%), test (10%)
splits = dataset.train_test_split(test_size=0.2, seed=42)
test_val = splits["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train": splits["train"],
    "validation": test_val["train"],
    "test": test_val["test"]
}).save_to_disk("patch_jigsaw_data")

# dataset_dict.push_to_hub("JXZhou0224/PatchJigsaw")