import os
import glob
from tqdm import tqdm


def quick_check_corrupted(directory):
    corrupted = []
    files = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

    print(f"  Found {len(files)} files, checking...")

    for file_path in tqdm(files, desc="    Scanning", unit="file"):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(20)

            if header.startswith(b'\xff\xd8'):
                f.seek(-2, 2)
                end = f.read(2)
                if end != b'\xff\xd9':
                    corrupted.append((file_path, "Missing JPEG end marker"))
            elif header.startswith(b'\x89PNG'):
                if len(header) < 8:
                    corrupted.append((file_path, "PNG too short"))
            else:
                corrupted.append((file_path, "Unknown format"))

        except Exception as e:
            corrupted.append((file_path, str(e)))

    return corrupted


def inspect_corrupted_full(directory):
    """Полная проверка через OpenCV (медленнее, но точнее)"""
    corrupted = []
    files = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

    print(f"  Found {len(files)} files, checking...")

    import cv2
    for file_path in tqdm(files, desc="    Scanning", unit="file"):
        try:
            img = cv2.imread(file_path)
            if img is None:
                corrupted.append((file_path, "Cannot read"))
        except Exception as e:
            corrupted.append((file_path, str(e)))

    return corrupted


if __name__ == "__main__":
    directories = [
        "D:/Datasets/RealAndSyntheticImages/train/real",
        "D:/Datasets/RealAndSyntheticImages/train/fake",
        "D:/Datasets/RealAndSyntheticImages/test/real",
        "D:/Datasets/RealAndSyntheticImages/test/fake"
    ]

    all_corrupted = []

    for d in directories:
        if os.path.exists(d):
            print(f"\nScanning {d}...")
            corrupted = quick_check_corrupted(d)
            all_corrupted.extend(corrupted)
            print(f"  Found {len(corrupted)} corrupted files in {d}")

    print(f"\nTotal corrupted: {len(all_corrupted)}")

    if all_corrupted:
        with open("corrupted_files_list.txt", "w") as f:
            for path, err in all_corrupted[:100]:
                f.write(f"{path}\t{err}\n")
        print("First 100 paths saved to corrupted_files_list.txt")