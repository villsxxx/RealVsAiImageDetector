import cv2
import torch
from torchvision import transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, img_height, img_width, is_train):
        self.pairs = pairs
        self.img_height = img_height
        self.img_width = img_width
        self.is_train = is_train

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.CenterCrop(224),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
