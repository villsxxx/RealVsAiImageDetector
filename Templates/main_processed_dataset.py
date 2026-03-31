import cv2
import glob
import os

ORIG_ROOT = "D:/Datasets/RealAndSyntheticImages"
TARGET_ROOT = "D:/Datasets/RealAndSyntheticImagesProcessed"
ERROR_LOG = os.path.join(TARGET_ROOT, "corrupted_files.txt")
TARGET_SIZE = (256, 256)

patterns = [
    "D:/Datasets/RealAndSyntheticImages/RealArt/RealArt/*",
    "D:/Datasets/RealAndSyntheticImages/train/real/*",
    "D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/real/train*/*",
    "D:/Datasets/RealAndSyntheticImages/AiArtData/AiArtData/*",
    "D:/Datasets/RealAndSyntheticImages/train/fake/*",
    "D:/Datasets/RealAndSyntheticImages/MNW/AI_Images/*/*",
    "D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/generated/train*/*",
    "D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/real/validation*/*",
    "D:/Datasets/RealAndSyntheticImages/test/real/*",
    "D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/generated/validation*/*",
    "D:/Datasets/RealAndSyntheticImages/test/fake/*"
]

os.makedirs(TARGET_ROOT, exist_ok=True)
error_log = open(ERROR_LOG, "w", encoding="utf-8")
success = 0
fail = 0

for pattern in patterns:
    files = glob.glob(pattern, recursive=False)
    print(f"Processing {pattern} -> {len(files)} files")
    for src_path in files:
        img = cv2.imread(src_path)
        if img is None:
            error_log.write(src_path + "\n")
            fail += 1
            continue

        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        rel_path = os.path.relpath(src_path, ORIG_ROOT)
        dst_path = os.path.join(TARGET_ROOT, rel_path)

        # Проверяем расширение и заменяем на .jpg, если не поддерживается
        ext = os.path.splitext(dst_path)[1].lower()
        supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        if ext not in supported_exts:
            # Если расширение не поддерживается или его нет, добавляем .jpg
            dst_path = os.path.splitext(dst_path)[0] + '.jpg'

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        success_flag = cv2.imwrite(dst_path, img)
        if not success_flag:
            error_log.write(f"Write failed: {src_path} -> {dst_path}\n")
            fail += 1
        else:
            success += 1

error_log.close()
print(f"Done. Success: {success}, Failed: {fail}. Log: {ERROR_LOG}")