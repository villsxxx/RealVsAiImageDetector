from Dataset.Dataset import CustomDataset
import glob

real_images_paths = sorted(glob.glob('D:/Datasets/RealAndSyntheticImages/RealArt/RealArt/*'))
generate_images_paths = sorted(glob.glob('D:/Datasets/RealAndSyntheticImages/AiArtData/AiArtData/*'))

pairs = []
for real_image in real_images_paths:
    pairs.append((real_image, 0))
for generate_image in generate_images_paths:
    pairs.append((generate_image, 1))

dataset = CustomDataset(pairs)