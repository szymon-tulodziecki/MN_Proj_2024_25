import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.transforms import transforms
from torchvision.ops import nms
import timeit
import random

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

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    elif backbone == 'mobilenetv2':
        backbone = torchvision.models.mobilenet_v2(weights=None).features
        backbone.out_channels = 1280

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    elif backbone == 'efficientnetb0':
        backbone = torchvision.models.efficientnet_b0(weights=None).features
        backbone.out_channels = 1280

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    elif backbone == 'vgg16':
        backbone = torchvision.models.vgg16(weights=None).features
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    else:
        raise ValueError("Unsupported backbone")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_random_color():
    return (random.random(), random.random(), random.random(), 0.4)

num_classes = 2
backbones = ['resnet50', 'resnet101', 'mobilenetv2', 'efficientnetb0', 'vgg16']
models = {backbone: get_model(num_classes, backbone) for backbone in backbones}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for backbone in backbones:
    models[backbone].load_state_dict(torch.load(f'faster_rcnn_{backbone}_model.pth'))
    models[backbone].eval()
    models[backbone].to(device)

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
    results = []

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

            confidence_threshold = 0.75  
            high_confidence_idxs = scores > confidence_threshold
            scores = scores[high_confidence_idxs].cpu().numpy()
            boxes = boxes[high_confidence_idxs].cpu().numpy()

            all_scores.extend(scores)

            results.append((image, img_name, boxes, scores))

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_score = np.mean(all_scores) if all_scores else 0
    std_score = np.std(all_scores) if all_scores else 0
    return avg_time, std_time, avg_score, std_score, results

evaluation_results = {}
for backbone in backbones:
    evaluation_results[backbone] = evaluate_model(models[backbone], test_loader, device)

print("\nCzasy predykcji:")
for backbone, (avg_time, std_time, _, _, _) in evaluation_results.items():
    print(f"{backbone}: Średni czas: {avg_time:.4f} sekund, Odchylenie standardowe: {std_time:.4f}")

print("\nWartości predykcji:")
for backbone, (_, _, avg_score, std_score, _) in evaluation_results.items():
    print(f"{backbone}: Średnia wartość: {avg_score:.4f}, Odchylenie standardowe: {std_score:.4f}")

for img_name in test_dataset.imgs:
    fig, axs = plt.subplots(1, len(backbones), figsize=(25, 5))
    
    if len(backbones) == 1:
        axs = [axs]
    
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    img_np = np.array(image)
    
    for ax, backbone in zip(axs, backbones):
        ax.imshow(img_np)
        
        for image, img_name_result, boxes, scores in evaluation_results[backbone][4]:
            if img_name == img_name_result:
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=get_random_color(), facecolor=get_random_color())
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f'{score:.2f}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.75))
        
        ax.set_title(f"{backbone}")
    
    fig.suptitle(f"Wyniki dla {img_name}")
    plt.show()
