import os
import glob


def inspect_first_files(directory, num=10):
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

    print(f"First {num} files in {directory}:")
    for f in files[:num]:
        print(f"  {f}")
        try:
            with open(f, 'rb') as file:
                header = file.read(20)
            print(f"    Header: {header[:20]}")
        except:
            pass


if __name__ == "__main__":
    inspect_first_files("D:/Datasets/RealAndSyntheticImages/train/real")
    inspect_first_files("D:/Datasets/RealAndSyntheticImages/train/fake")
    inspect_first_files("D:/Datasets/RealAndSyntheticImages/test/real")
    inspect_first_files("D:/Datasets/RealAndSyntheticImages/test/fake")