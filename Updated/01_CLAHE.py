import os
import cv2
import numpy as np
from tqdm import tqdm
# ================================
# Config
# ================================
INPUT_FOLDER = './Raw Dataset'
OUTPUT_FOLDER = './CLAHE'
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
# ================================

# Collect all image file paths (recursive)
image_files = []
for root, _, files in os.walk(INPUT_FOLDER):
    for f in files:
        if f.lower().endswith(IMAGE_EXTENSIONS):
            rel_path = os.path.relpath(root, INPUT_FOLDER)  # relative subfolder
            image_files.append((os.path.join(root, f), rel_path, f))

# Process each image
# Process each image
for img_path, rel_path, filename in tqdm(image_files, desc="Enhancing Images", ncols=80):
    # Create corresponding output folder
    output_subfolder = os.path.join(OUTPUT_FOLDER, rel_path)
    os.makedirs(output_subfolder, exist_ok=True)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"⚠️ Skipping unreadable image: {filename}")
        continue

    # Fixed Parameters for Medical Imaging (Chest X-Ray)
    # Clip Limit 2.0: Prevents noise amplification in flat regions
    # Tile Size 8x8: Standard local contrast enhancement window
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Save enhanced image in corresponding subfolder
    save_path = os.path.join(output_subfolder, filename)
    cv2.imwrite(save_path, enhanced)
