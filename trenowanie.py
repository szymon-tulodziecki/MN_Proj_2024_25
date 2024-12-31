import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator
from sklearn.model_selection import train_test_split

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
                x_min, y_min, szerokosc, wysokosc = anno['bbox']
                x_max, y_max = x_min + szerokosc, y_min + wysokosc
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

def get_model(num_classes, backbone='resnet50'):
    if backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif backbone == 'resnet101':
        backbone = torchvision.models.resnet101(weights="DEFAULT")
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048

        anchor_sizes = ((32, 64, 128, 256, 512),)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    else:
        raise ValueError("Unsupported backbone")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

num_classes = 2

model_resnet50 = get_model(num_classes, backbone='resnet50')
model_resnet101 = get_model(num_classes, backbone='resnet101')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_resnet50 = model_resnet50.to(device)
model_resnet101 = model_resnet101.to(device)

import torch.optim as optim

def train_one_epoch(model, optimizer, data_loader, device):
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

optimizer_resnet50 = optim.SGD(model_resnet50.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_resnet101 = optim.SGD(model_resnet101.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 5

for epoch in range(num_epochs):
    train_loss_resnet50 = train_one_epoch(model_resnet50, optimizer_resnet50, train_loader, device)
    train_loss_resnet101 = train_one_epoch(model_resnet101, optimizer_resnet101, train_loader, device)
    print(f"Epoka [{epoch+1}/{num_epochs}], Strata ResNet-50: {train_loss_resnet50:.4f}, Strata ResNet-101: {train_loss_resnet101:.4f}")

model_save_path_resnet50 = 'faster_rcnn_resnet50_model.pth'
model_save_path_resnet101 = 'faster_rcnn_resnet101_model.pth'
torch.save(model_resnet50.state_dict(), model_save_path_resnet50)
torch.save(model_resnet101.state_dict(), model_save_path_resnet101)
print(f"Model ResNet-50 został zapisany w: {model_save_path_resnet50}")
print(f"Model ResNet-101 został zapisany w: {model_save_path_resnet101}")
