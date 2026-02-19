import torch
import torchvision
import torchvision.transforms.functional as F
import cv2


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_pairs):
        self.dataset_pairs = dataset_pairs

    def __len__(self):
        return len(self.dataset_pairs)

    def __getitem__(self, item):
        image_path = self.dataset_pairs[item][0]
        class_item = self.dataset_pairs[item][1]

        image_item = cv2.imread(image_path)
        image_item = cv2.cvtColor(image_item, cv2.COLOR_BGR2RGB)
        image_item = torchvision.transforms.ToTensor()(image_item)
        image_item = torchvision.transforms.functional.resize(image_item, [256, 256])
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(image_item, output_size=(224, 224))
        image_item = torchvision.transforms.functional.crop(image_item, i, j, h, w)
        if torch.rand(1).item() > 0.5:
            image_item = torchvision.transforms.functional.hflip(image_item)
        image_item = torchvision.transforms.functional.normalize(image_item, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image_item, torch.tensor(class_item, dtype=torch.long)