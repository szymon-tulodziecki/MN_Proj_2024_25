import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.ops import nms

class CustomTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [img for img in sorted(os.listdir(img_dir)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.imgs[idx]

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load('faster_rcnn_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dir = 'test'
test_dataset = CustomTestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for images, img_names in test_loader:
    images = list(img.to(device) for img in images)

    with torch.no_grad():
        predictions = model(images)

    for image, img_name, prediction in zip(images, img_names, predictions):
        boxes = prediction['boxes']
        scores = prediction['scores']

        keep = nms(boxes, scores, iou_threshold=0.4)
        boxes = boxes[keep]
        scores = scores[keep]

        confidence_threshold = 0.75
        high_confidence_idxs = scores > confidence_threshold
        boxes = boxes[high_confidence_idxs].cpu().numpy()
        scores = scores[high_confidence_idxs].cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        img_np = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=12)

        plt.title(f"Wyniki dla {img_name}")
        plt.show()