import torch
import os
from PIL import Image
from torchvision import transforms
from Model.ResNet import CustomResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

inference_image_path = 'D:/Validation_example/real_photo.jpg'

model = CustomResNet(num_classes=2).to(device)

model_path = 'D:/nnModels/RGDetector/run_1/best_model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

try:
    image = Image.open(inference_image_path).convert('RGB')
except Exception as e:
    print(f"Ошибка при открытии изображения: {e}")
    exit()

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    confidence = probabilities[predicted_class].item()

class_names = {0: "Real Image (Настоящее)", 1: "AI Generated (Сгенерированное)"}

print("-" * 30)
print(f"Изображение: {os.path.basename(inference_image_path)}")
print(f"Предсказанный класс: {class_names[predicted_class]}")
print(f"Уверенность модели: {confidence * 100:.2f}%")
print("-" * 30)