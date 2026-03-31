import pandas as pd
import os
import glob
from PIL import Image
import io


def extract_parquet_to_images(parquet_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    for pq_file in parquet_files:
        print(f"\nProcessing {os.path.basename(pq_file)}...")
        df = pd.read_parquet(pq_file)

        model_name = os.path.splitext(os.path.basename(pq_file))[0]

        saved_count = 0
        for idx, row in df.iterrows():
            label = row['label']
            class_dir = "real" if label == 0 else "generated"

            save_dir = os.path.join(output_dir, class_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)

            image_data = row['image']
            img_bytes = image_data['bytes']

            img_path = os.path.join(save_dir, f"{idx:06d}.jpg")

            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(img_path, 'JPEG', quality=95)
                saved_count += 1
            except Exception as e:
                print(f"  Error at index {idx}: {e}")
        print(f"  Saved {saved_count}/{len(df)} images from {model_name}")

    print("\nExtraction complete!")
if __name__ == "__main__":
    extract_parquet_to_images(
        parquet_dir="D:/Datasets/synthbuster-plus/data",
        output_dir="D:/Datasets/synthbuster-plus_images"
    )