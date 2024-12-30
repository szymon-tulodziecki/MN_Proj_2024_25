import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.model_selection import train_test_split

"""
Na podstawie: 
- https://pytorch.org/docs/stable/index.html
- https://pytorch.org/vision/stable/index.html
- https://www.youtube.com/watch?v=4OXntFVfFio
- https://www.youtube.com/watch?v=xYd95gppJ-0
"""
class CustomDataset(Dataset):
    """Tworzenie dataset'u zgodnie z https://www.geeksforgeeks.org/image-datasets-dataloaders-and-transforms-in-pytorch/"""
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data[idx]['image'])
        image = Image.open(img_name).convert("RGB")
        annotations = self.data[idx]['annotations']

        boxes = []
        labels = []
        for anno in annotations:
            if anno['class'] == 'samochod':
                x_min, y_min, x_max, y_max = anno['bbox']
                x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                y_min, y_max = min(y_min, y_max), max(y_min, y_max)
                if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.data))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transform:
            image = self.transform(image)

        return image, target

json_path = 'adnotacje.json'
img_dir = 'dodane'

with open(json_path, 'r', encoding='utf-8') as f:
    all_data = json.load(f)

train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CustomDataset(train_data, img_dir, transform=transform)
val_dataset = CustomDataset(val_data, img_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

num_classes = 2
model = get_model(num_classes)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

import torch.optim as optim

def train_one_epoch(model, optimizer, data_loader, device):
    """Funkcja trenująca model na podstawie danych z https://pytorch.org/vision/stable/models.html#faster-r-cnn"""
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 5

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    print(f"Epoka [{epoch+1}/{num_epochs}], Strata: {train_loss:.4f}")

model_save_path = 'faster_rcnn_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model został zapisany w: {model_save_path}")