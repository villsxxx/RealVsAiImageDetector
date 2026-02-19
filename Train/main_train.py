from Dataset.Dataset import CustomDataset
from Model.ResNet import CustomResNet
import glob
import torch
import os
from tqdm import tqdm

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
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = CustomResNet(num_classes=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_train_loss = float('inf')
os.makedirs('D:/nnModels/RGDetector', exist_ok=True)

NUM_EPOCHS = 1000

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0

    pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')

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
    train_acc = 100 * train_correct / len(dataset)

    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), 'D:/nnModels/RGDetector/best_model.pth')
        print(f"  ✓ Модель сохранена!")

print('\nSuccess')