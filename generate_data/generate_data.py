import os, csv, json, random
from pathlib import Path
from PIL import Image
from .process_image import create_jigsaw,FACTOR

INPUT_DIR  = Path("data/coco_train")
OUTPUT_DIR = Path("patch_jigsaw")
CSV_PATH   = Path("jigsaw_data.csv")

OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

def sample_params():
    return {
        "h_div": random.choice([2,3,4]),
        "num_of_swap": random.choice([1,2,3])
    }

with CSV_PATH.open("w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["id","image","h_div","num_of_swap","swap_log","grid_w","grid_h","factor","source_image"])
    idx = 1

    for img_path in sorted(INPUT_DIR.iterdir()):
        if not img_path.suffix.lower() in [".jpg",".jpeg",".png"]:
            continue
        params = sample_params()
        try:
            img = Image.open(img_path)
            jig, swap_log = create_jigsaw(
                img,
                h_div=params["h_div"],
                num_of_swap=params["num_of_swap"],
            )
        except Exception as e:
            print("unable to process image")
            print(img_path)
            # skip images that can’t be jigsawed
            continue

        grid_h = jig.height // FACTOR
        grid_w = jig.width  // FACTOR

        out_name = f"{img_path.stem}_f{FACTOR}_h{params['h_div']}_s{params['num_of_swap']}.png"
        out_name =  "_".join(out_name.split("_")[2:])
        out_path = OUTPUT_DIR / out_name
        jig.save(out_path, format="PNG")

        writer.writerow([
            idx,
            str(out_name),
            params["h_div"],
            params["num_of_swap"],
            json.dumps(swap_log),
            grid_w,
            grid_h, 
            FACTOR,
            img_path.name,          
        ])
        idx += 1

print(f"Generated {idx-1} jigsaws. CSV at {CSV_PATH}")