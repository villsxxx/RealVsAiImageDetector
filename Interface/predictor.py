import torch
import torchvision.transforms as transforms
from PIL import Image
import os

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'ActualModels', 'best-epoch=91-val_loss=0.4410.ckpt')
CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_PATH)


from Models import CustomResNet


class ImagePredictor:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomResNet(num_classes=2).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_class1 = probabilities[0, 1].item()
            predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class, confidence_class1

predictor = ImagePredictor()