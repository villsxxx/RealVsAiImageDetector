import glob

real_val = (
    glob.glob("D:/Datasets/RealAndSyntheticImages/RealArt/RealArt/*") +
    glob.glob("D:/Datasets/RealAndSyntheticImages/test/real/*") +
    glob.glob("D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/real/validation*/*")
)

fake_val = (
    glob.glob("D:/Datasets/RealAndSyntheticImages/AiArtData/AiArtData/*") +
    glob.glob("D:/Datasets/RealAndSyntheticImages/test/fake/*") +
    glob.glob("D:/Datasets/RealAndSyntheticImages/synthbuster-plus_images/generated/validation*/*")
)

print(f"Real validation files: {len(real_val)}")
print(f"Fake validation files: {len(fake_val)}")