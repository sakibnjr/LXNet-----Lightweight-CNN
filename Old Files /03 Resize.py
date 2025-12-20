import os
import cv2
from tqdm import tqdm

INPUT_FOLDER = "./CLAHE9"
OUTPUT_FOLDER = "./Resized9"
TARGET_SIZE = (224, 224)

def resize_images_in_folder(input_dir, output_dir, size=(224, 224), jpeg_quality=100):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []

    for folder, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_input_path = os.path.join(folder, file)
                rel_path = os.path.relpath(full_input_path, input_dir)

                rel_path_jpeg = os.path.splitext(rel_path)[0] + ".jpg"
                full_output_path = os.path.join(output_dir, rel_path_jpeg)

                image_paths.append((full_input_path, full_output_path))

    for input_path, output_path in tqdm(image_paths, desc="Resizing & Saving as JPEG"):
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"Warning: Skipping unreadable file {input_path}")
                continue

            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    if os.path.isdir(INPUT_FOLDER):
        resize_images_in_folder(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_SIZE, jpeg_quality=100)
        print("✅ Resizing and conversion to JPEG complete.")
    else:
        print("❌ Invalid input folder path.")
