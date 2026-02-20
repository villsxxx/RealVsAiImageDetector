from Dataset.Dataset import CustomDataset
from Model.ResNet import CustomResNet
import glob
import torch
import os
from tqdm import tqdm
from torch.utils.data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

real_images_paths = sorted(glob.glob('D:/Datasets/RealAndSyntheticImages/RealArt/RealArt/*'))
generate_images_paths = sorted(glob.glob('D:/Datasets/RealAndSyntheticImages/AiArtData/AiArtData/*'))

pairs = []
for real_image in real_images_paths:
    pairs.append((real_image, 0))
for generate_image in generate_images_paths:
    pairs.append((generate_image, 1))

print(f"Всего изображений: {len(pairs)}")

dataset = CustomDataset(pairs)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

model = CustomResNet(num_classes=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float('inf')
os.makedirs('D:/nnModels/RGDetector', exist_ok=True)

epochs = 1000

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{epochs}]')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / len(train_dataset)

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / len(val_dataset)

    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'D:/nnModels/RGDetector/best_model.pth')
        print(f"  ✓ Модель сохранена!")

print('\nSuccess')