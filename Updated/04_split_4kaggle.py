import os
import random
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------
data_dir = Path("/kaggle/input/9classresized")
output_dir = Path("/kaggle/working/splited")
train_ratio = 0.8 
val_ratio = 0.1
test_ratio = 0.1
seed = 42

random.seed(seed) 

# Clean output dir if it exists to avoid mixing old runs
if output_dir.exists():
    import shutil
    shutil.rmtree(output_dir)

classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
print(f"Found classes: {classes}")

for cls in classes:
    class_dir = data_dir / cls
    images = sorted(list(class_dir.glob("*"))) # Sort first to ensure deterministic shuffle with seed
    random.shuffle(images)

    n_total = len(images)
    if n_total == 0:
        print(f"[{cls}] Skipping (empty folder)")
        continue

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remainder goes to test to ensure total matches

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    print(f"[{cls}] Total: {n_total} -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

    for split, files in splits.items():
        split_dir = output_dir / split / cls
        split_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            # OPTION A: Symlink (Fast, saves space, good for training in same session)
            os.symlink(f, split_dir / f.name)
            
            # OPTION B: Copy (Use only if you plan to download the output folder as a zip later)
            # shutil.copy(f, split_dir / f.name)

print(f"\n✅ Splitting Complete. Data located at: {output_dir}")