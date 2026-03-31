import os
import glob
import cv2
from PIL import Image

ORIG_ROOT = "D:/Datasets/RealAndSyntheticImages"
TARGET_ROOT = "D:/Datasets/RealAndSyntheticImagesProcessedAfterPillow"
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
        # Проверка целостности через Pillow
        try:
            with Image.open(src_path) as img:
                img.verify()          # Проверка структуры файла
        except Exception as e:
            error_log.write(f"{src_path} : {e}\n")
            fail += 1
            continue

        # Читаем через OpenCV (изображение гарантированно читается)
        img_cv = cv2.imread(src_path)
        if img_cv is None:
            error_log.write(f"{src_path} : OpenCV failed to read\n")
            fail += 1
            continue

        # Опциональный ресайз
        img_cv = cv2.resize(img_cv, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Формируем путь для сохранения
        rel_path = os.path.relpath(src_path, ORIG_ROOT)
        dst_path = os.path.join(TARGET_ROOT, rel_path)
        # Для единообразия сохраняем как JPEG
        dst_path = os.path.splitext(dst_path)[0] + ".jpg"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        cv2.imwrite(dst_path, img_cv)
        success += 1

error_log.close()
print(f"Done. Success: {success}, Failed: {fail}. Log: {ERROR_LOG}")