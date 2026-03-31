checkpoint_path = r"D:\GenerateImageRealImageDetector\RealVsAiImageDetector\ActualModels\best-epoch=22-val_loss=0.2484.ckpt"

with open(checkpoint_path, 'rb') as f:
    header = f.read(100)
    print(header)