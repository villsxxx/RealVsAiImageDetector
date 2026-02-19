import torch
import torchvision
import cv2

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_pairs):
        self.dataset_pairs = dataset_pairs

    def __len__(self):
        return  len(self.dataset_pairs)

    def __getitem__(self, item):
        image_item = self.dataset_pairs[item][0]
        class_item = self.dataset_pairs[item][0]
        return image_item, class_item

