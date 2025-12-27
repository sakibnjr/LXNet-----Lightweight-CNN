import os
import random
import shutil
from pathlib import Path
from typing import List
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

# ========= HARD-CODED PATHS =========
INPUT_DIR = Path("./Dataset/01_CLAHE")
OUTPUT_DIR = Path("./Dataset/02_AUGMENTED_BALANCED/")
# ====================================

# ========= SETTINGS =========
RANDOM_SEED = 42
TARGET_PER_CLASS = 1000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CLEAN_OUTPUT_CLASS_DIRS = True

# --- Medically Safe Intensities ---
ROTATION_RANGE = (-5.0, 5.0)       # Degrees
ZOOM_RANGE = (1.05, 1.15)          # Zoom in
TRANSLATION_FACTOR = 0.05          # Max shift 5%

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

def save_image(img: Image.Image, out_path: Path):
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(out_path, quality=95, subsampling=0)
    else:
        img.save(out_path)

def get_median_color(img: Image.Image) -> int:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return int(np.median(arr))

# ========= SAFE AUGMENTATION FUNCTIONS =========

def aug_rotate(img: Image.Image) -> Image.Image:
    angle = random.uniform(*ROTATION_RANGE)
    fill = get_median_color(img)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill)

def aug_zoom(img: Image.Image) -> Image.Image:
    factor = random.uniform(*ZOOM_RANGE)
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)

    img_zoomed = img.resize((new_w, new_h), resample=Image.BILINEAR)
    left = (new_w - w) / 2
    top = (new_h - h) / 2
    return img_zoomed.crop((left, top, left + w, top + h))

def aug_translate(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_dx = w * TRANSLATION_FACTOR
    max_dy = h * TRANSLATION_FACTOR
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)

    fill = get_median_color(img)
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=fill
    )

def aug_hflip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)

# ========= AUGMENTATION POOL =========
AUG_FUNCTIONS = [
    (aug_rotate, "rot"),
    (aug_zoom, "zoom"),
    (aug_translate, "trans"),
    (aug_hflip, "hflip")
]

def copy_files(paths: List[Path], dst_dir: Path, desc: str):
    for p in tqdm(paths, desc=desc, unit="img"):
        dst = dst_dir / p.name
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

    class_to_imgs = {cdir.name: list_images(cdir) for cdir in class_dirs}
    class_to_imgs = {k: v for k, v in class_to_imgs.items() if v}

    print("Input class counts:")
    for cls, imgs in class_to_imgs.items():
        print(f"  - {cls}: {len(imgs)}")

    for cls, imgs in class_to_imgs.items():
        out_dir = OUTPUT_DIR / cls
        ensure_dir(out_dir)
        if CLEAN_OUTPUT_CLASS_DIRS:
            empty_dir(out_dir)

        n = len(imgs)
        print(f"\n[{cls}] Input: {n} -> Target: {TARGET_PER_CLASS}")

        if n >= TARGET_PER_CLASS:
            selected = random.sample(imgs, TARGET_PER_CLASS)
            copy_files(selected, out_dir, desc=f"Copying {cls}")
            continue

        copy_files(imgs, out_dir, desc=f"Copying Originals")

        need = TARGET_PER_CLASS - n
        aug_idx = 0

        for _ in tqdm(range(need), desc=f"Augmenting {cls}", unit="img"):
            src_path = random.choice(imgs)
            with Image.open(src_path) as im:
                if im.mode == "P":
                    im = im.convert("RGB")

                aug_func, aug_tag = random.choice(AUG_FUNCTIONS)
                aug_img = aug_func(im)

                out_name = f"{src_path.stem}__aug_{aug_tag}_{aug_idx:05d}{src_path.suffix}"
                out_path = out_dir / out_name

                while out_path.exists():
                    aug_idx += 1
                    out_path = out_dir / f"{src_path.stem}__aug_{aug_tag}_{aug_idx:05d}{src_path.suffix}"

                save_image(aug_img, out_path)
                aug_idx += 1

        print(f"[{cls}] Done. Final count: {len(list_images(out_dir))}")

    print(f"\nAll classes processed. Output at: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
