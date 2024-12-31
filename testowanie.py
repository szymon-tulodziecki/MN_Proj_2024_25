import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.transforms import transforms
from torchvision.ops import nms
import timeit

class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [img for img in sorted(os.listdir(img_dir)) if img.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.imgs[idx]

def get_model(num_classes, backbone='resnet50'):
    if backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)  
    elif backbone == 'resnet101':
        backbone = torchvision.models.resnet101(weights=None)  
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048

        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    else:
        raise ValueError("Unsupported backbone")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

num_classes = 2

model_resnet50 = get_model(num_classes, backbone='resnet50')
model_resnet101 = get_model(num_classes, backbone='resnet101')

model_resnet50.load_state_dict(torch.load('faster_rcnn_resnet50_model.pth'))
model_resnet101.load_state_dict(torch.load('faster_rcnn_resnet101_model.pth'))

model_resnet50.eval()
model_resnet101.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_resnet50.to(device)
model_resnet101.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dir = 'test'
test_dataset = CustomTestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def evaluate_model(model, test_loader, device):
    model.eval()
    times = []
    all_scores = []

    for images, img_names in test_loader:
        images = list(img.to(device) for img in images)

        start_time = timeit.default_timer()
        with torch.no_grad():
            predictions = model(images)
        end_time = timeit.default_timer()
        inference_time = end_time - start_time
        times.append(inference_time)

        for image, img_name, prediction in zip(images, img_names, predictions):
            boxes = prediction['boxes']
            scores = prediction['scores']

            keep = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep]
            scores = scores[keep]

            confidence_threshold = 0.75 #ustawiamy próg pewności (żeby uniknąć wyświetlania zbyt wielu bounding boxów)
            high_confidence_idxs = scores > confidence_threshold
            scores = scores[high_confidence_idxs].cpu().numpy()

            all_scores.extend(scores)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_score = np.mean(all_scores) if all_scores else 0
    std_score = np.std(all_scores) if all_scores else 0
    return avg_time, std_time, avg_score, std_score

avg_time_resnet50, std_time_resnet50, avg_score_resnet50, std_score_resnet50 = evaluate_model(model_resnet50, test_loader, device)
avg_time_resnet101, std_time_resnet101, avg_score_resnet101, std_score_resnet101 = evaluate_model(model_resnet101, test_loader, device)

print("\n\n")
print(f"ResNet-50: Średni czas predykcji: {avg_time_resnet50:.4f} sekund, Odchylenie standardowe: {std_time_resnet50:.4f}")
print(f"ResNet-50: Średnia wartość predykcji: {avg_score_resnet50:.4f}, Odchylenie standardowe tej wartości: {std_score_resnet50:.4f}")
print(f"ResNet-101: Średni czas predykcji: {avg_time_resnet101:.4f} sekund, Odchylenie standardowe: {std_time_resnet101:.4f}")
print(f"ResNet-101: Średnia wartość predykcji: {avg_score_resnet101:.4f}, Odchylenie standardowe tej wartości: {std_score_resnet101:.4f}")
print("\n\n")

for (image_resnet50, img_name_resnet50, boxes_resnet50, scores_resnet50), (image_resnet101, img_name_resnet101, boxes_resnet101, scores_resnet101) in zip(results_resnet50, results_resnet101):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    ax = axs[0]
    img_np = image_resnet50.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img_np)
    for box, score in zip(boxes_resnet50, scores_resnet50):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=12)
    ax.set_title(f"Wyniki ResNet-50 dla {img_name_resnet50}")

    ax = axs[1]
    img_np = image_resnet101.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img_np)
    for box, score in zip(boxes_resnet101, scores_resnet101):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=12)
    ax.set_title(f"Wyniki ResNet-101 dla {img_name_resnet101}")

    plt.show()
