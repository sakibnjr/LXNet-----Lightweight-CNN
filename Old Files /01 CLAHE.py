import os
import cv2
import numpy as np
from tqdm import tqdm
import itertools

# ================================
# Config
# ================================
INPUT_FOLDER = './9 class'
OUTPUT_FOLDER = './CLAHE9'
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
CLIP_LIMITS = np.arange(1.0, 5.1, 0.5)       
TILE_SIZES = list(range(2, 17))              
# ================================

# Collect all image file paths (recursive)
image_files = []
for root, _, files in os.walk(INPUT_FOLDER):
    for f in files:
        if f.lower().endswith(IMAGE_EXTENSIONS):
            rel_path = os.path.relpath(root, INPUT_FOLDER)  # relative subfolder
            image_files.append((os.path.join(root, f), rel_path, f))

# Process each image
for img_path, rel_path, filename in tqdm(image_files, desc="Enhancing Images (Grid Search)", ncols=80):
    # Create corresponding output folder
    output_subfolder = os.path.join(OUTPUT_FOLDER, rel_path)
    os.makedirs(output_subfolder, exist_ok=True)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"⚠️ Skipping unreadable image: {filename}")
        continue

    best_contrast = -np.inf
    best_params = None
    best_enhanced = None

    # Grid search: try all combinations
    for clip, tile in itertools.product(CLIP_LIMITS, TILE_SIZES):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        enhanced = clahe.apply(image)
        contrast = enhanced.std()

        if contrast > best_contrast:
            best_contrast = contrast
            best_params = (clip, tile)
            best_enhanced = enhanced

    # Save best-enhanced image in corresponding subfolder
    save_path = os.path.join(output_subfolder, filename)
    cv2.imwrite(save_path, best_enhanced)

    # Optional logging
    print(f"[{os.path.join(rel_path, filename)}] ➤ clip={best_params[0]:.2f}, tile={best_params[1]}, contrast={best_contrast:.2f}")
