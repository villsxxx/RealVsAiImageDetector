import os
import cv2
import torch
import torchvision
import torchvision.transforms.functional as F
import glob
dataset_path = "D:/Datasets/RealAndSyntheticImages/"

for image_path in glob.glob(os.path.join(dataset_path, "*/*/*")):
    try:
        image_item = cv2.imread(image_path)
        image_item = cv2.cvtColor(image_item, cv2.COLOR_BGR2RGB)
        image_item = torchvision.transforms.ToTensor()(image_item)
        image_item = F.resize(image_item, [256, 256])
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(image_item, output_size=(224, 224))
        image_item = F.crop(image_item, i, j, h, w)
        if torch.rand(1).item() > 0.5:
            image_item = F.hflip(image_item)
        image_item = F.normalize(image_item, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    except Exception as e:
        print(f"Corrupt: {image_path} | {e}")