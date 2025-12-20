import os
import shutil
import random
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------
data_dir = "/kaggle/input/9classresized"          
output_dir = "/kaggle/working/splited"  
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
seed = 42 

random.seed(seed)
os.makedirs(output_dir, exist_ok=True)

classes = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]

for cls in classes:
    class_dir = Path(data_dir) / cls
    images = list(class_dir.glob("*"))

    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, files in splits.items():
        split_dir = Path(output_dir) / split / cls
        split_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, split_dir / f.name)

    print(f"[{cls}] -> Train: {n_train}, Val: {n_val}, Test: {n_test}") 