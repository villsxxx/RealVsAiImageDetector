import torch
import torchvision
import torchvision.transforms.functional as F
import cv2
from torchvision import transforms

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, img_height=256, img_width=256, crop_size=224,
                 normalize_mean=None, normalize_std=None,
                 use_horizontal_flip=True, is_train=True):
        self.pairs = pairs
        self.img_height = img_height
        self.img_width = img_width
        self.crop_size = crop_size
        self.use_horizontal_flip = use_horizontal_flip
        self.is_train = is_train
        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]
        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, label = self.pairs[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torchvision.transforms.ToTensor()(image)
        image = F.resize(image, [self.img_height, self.img_width])
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size)
            )
            image = F.crop(image, i, j, h, w)
            if self.use_horizontal_flip and torch.rand(1).item() > 0.5:
                image = F.hflip(image)
        else:
            image = F.center_crop(image, self.crop_size)
        image = F.normalize(image, mean=self.normalize_mean, std=self.normalize_std)
        return image, torch.tensor(label, dtype=torch.long)