import os
import cv2
import glob
from tqdm import tqdm


def inspect_corrupted(directory):
    corrupted = []
    files = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

    print(f"  Found {len(files)} files, checking...")

    for file_path in tqdm(files, desc=f"    Scanning", unit="file"):
        try:
            img = cv2.imread(file_path)
            if img is None:
                with open(file_path, 'rb') as f:
                    header = f.read(20)
                corrupted.append((file_path, "Cannot read (img is None)", header[:20]))
            else:
                try:
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    corrupted.append((file_path, f"Color conversion error: {e}", None))
        except Exception as e:
            corrupted.append((file_path, str(e), None))

    return corrupted


def show_corrupted_samples(corrupted, num_samples=20):
    print(f"\n{'=' * 80}")
    print(f"Found {len(corrupted)} corrupted files")
    print(f"Showing first {min(num_samples, len(corrupted))} files:\n")

    for i, (path, error, header) in enumerate(corrupted[:num_samples]):
        print(f"[{i + 1}] {path}")
        print(f"    Error: {error}")
        if header:
            print(f"    Header: {header}")
        print()


if __name__ == "__main__":
    directories = [
        "D:/Datasets/RealAndSyntheticImages/train/real",
        "D:/Datasets/RealAndSyntheticImages/train/fake",
        "D:/Datasets/RealAndSyntheticImages/test/real",
        "D:/Datasets/RealAndSyntheticImages/test/fake",
        # "D:/Datasets/RealAndSyntheticImages/MNW/AI_Images",
        # "D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/real",
        # "D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/generated"
    ]

    all_corrupted = []

    for d in directories:
        if os.path.exists(d):
            print(f"\nScanning {d}...")
            corrupted = inspect_corrupted(d)
            all_corrupted.extend(corrupted)
            print(f"  Found {len(corrupted)} corrupted files in {d}")

    show_corrupted_samples(all_corrupted, num_samples=30)

    with open("corrupted_files_list.txt", "w", encoding="utf-8") as f:
        for path, error, _ in all_corrupted:
            f.write(f"{path}\t{error}\n")

    print(f"\nFull list saved to corrupted_files_list.txt")

    if all_corrupted:
        print("\nWhat do you want to do?")
        print("1. Just show me the list (done)")
        print("2. Remove all corrupted files")
        print("3. Move corrupted files to backup folder")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "2":
            print("\nRemoving corrupted files...")
            for path, _, _ in tqdm(all_corrupted, desc="Removing"):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Failed to remove {path}: {e}")
            print(f"\nRemoved {len(all_corrupted)} files")

        elif choice == "3":
            backup_dir = "D:/Datasets/corrupted_backup"
            os.makedirs(backup_dir, exist_ok=True)
            print(f"\nMoving corrupted files to {backup_dir}...")
            for path, _, _ in tqdm(all_corrupted, desc="Moving"):
                try:
                    new_path = os.path.join(backup_dir, os.path.basename(path))
                    os.rename(path, new_path)
                except Exception as e:
                    print(f"Failed to move {path}: {e}")
            print(f"\nMoved {len(all_corrupted)} files to {backup_dir}")