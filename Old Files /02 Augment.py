import os
import random
import shutil
from pathlib import Path
from typing import List
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

# ========= HARD-CODED PATHS (edit these) =========
INPUT_DIR = Path("/path/to/chestxray/train")
OUTPUT_DIR = Path("/path/to/chestxray_1000each")
# ================================================

# Settings
RANDOM_SEED = 42
TARGET_PER_CLASS = 1000
ROTATION_RANGE_DEGREES = (-5.0, 5.0)  # small, "medical-safe" rotations for CXR
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CLEAN_OUTPUT_CLASS_DIRS = True  # if True, empties each OUTPUT_DIR/<class> before writing
random.seed(RANDOM_SEED)

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def empty_dir(d: Path):
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

def horizontal_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)

def random_rotation_small(img: Image.Image) -> Image.Image:
    angle = random.uniform(*ROTATION_RANGE_DEGREES)
    # Use grayscale only to compute a soft fill value (does NOT change output mode)
    arr = np.array(img.convert("L"), dtype=np.uint8)
    fill = int(np.median(arr))
    return img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=fill)

def save_image(img: Image.Image, out_path: Path):
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(out_path, quality=95, subsampling=0)
    else:
        img.save(out_path)

def copy_files(paths: List[Path], dst_dir: Path, desc: str):
    for p in tqdm(paths, desc=desc, unit="img"):
        dst = dst_dir / p.name
        # Avoid collisions by renaming if necessary
        if dst.exists():
            stem, ext = p.stem, p.suffix
            k = 1
            new_dst = dst_dir / f"{stem}__copy_{k:03d}{ext}"
            while new_dst.exists():
                k += 1
                new_dst = dst_dir / f"{stem}__copy_{k:03d}{ext}"
            dst = new_dst
        shutil.copy2(p, dst)

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR}")

    class_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under {INPUT_DIR}")

    # Gather images per class
    class_to_imgs = {cdir.name: list_images(cdir) for cdir in class_dirs}
    class_to_imgs = {k: v for k, v in class_to_imgs.items() if v}  # drop empty classes

    if not class_to_imgs:
        raise RuntimeError("No images found in any class subfolder.")

    print("Input class counts:")
    for cls, imgs in class_to_imgs.items():
        print(f"  - {cls}: {len(imgs)}")

    # Process each class to exactly TARGET_PER_CLASS
    for cls, imgs in class_to_imgs.items():
        out_dir = OUTPUT_DIR / cls
        ensure_dir(out_dir)
        if CLEAN_OUTPUT_CLASS_DIRS:
            empty_dir(out_dir)

        n = len(imgs)
        print(f"\n[{cls}] input count: {n}")

        if n == TARGET_PER_CLASS:
            # Just copy all originals
            print(f"[{cls}] Exactly {TARGET_PER_CLASS} images; copying originals…")
            copy_files(imgs, out_dir, desc=f"Copying {cls}")
            print(f"[{cls}] Done.")
            continue

        if n > TARGET_PER_CLASS:
            # Downsample to 1000 deterministically
            print(f"[{cls}] > {TARGET_PER_CLASS}; downsampling to {TARGET_PER_CLASS} (random with seed={RANDOM_SEED})…")
            # random.sample is reproducible given RANDOM_SEED above
            selected = random.sample(imgs, TARGET_PER_CLASS)
            copy_files(selected, out_dir, desc=f"Downsampling {cls}")
            print(f"[{cls}] Done.")
            continue

        # n < TARGET_PER_CLASS: copy originals, then augment to reach target
        print(f"[{cls}] < {TARGET_PER_CLASS}; copying originals then augmenting {TARGET_PER_CLASS - n} images…")
        copy_files(imgs, out_dir, desc=f"Copying {cls}")

        need = TARGET_PER_CLASS - n
        aug_idx = 0

        for _ in tqdm(range(need), desc=f"Augmenting {cls}", unit="img"):
            src_path = random.choice(imgs)
            with Image.open(src_path) as im:
                # Keep original mode (no grayscale conversion)
                if random.random() < 0.5:
                    aug_img = horizontal_flip(im)
                    aug_tag = "hflip"
                else:
                    aug_img = random_rotation_small(im)
                    aug_tag = "rot"

                stem = src_path.stem
                ext = src_path.suffix.lower()
                out_name = f"{stem}__aug_{aug_tag}_{aug_idx:05d}{ext}"
                out_path = out_dir / out_name

                # Ensure unique filename
                while out_path.exists():
                    aug_idx += 1
                    out_name = f"{stem}__aug_{aug_tag}_{aug_idx:05d}{ext}"
                    out_path = out_dir / out_name

                save_image(aug_img, out_path)
                aug_idx += 1

        final = len(list_images(out_dir))
        print(f"[{cls}] Done. Final count: {final}")

    print(f"\nAll classes processed. Output at: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
