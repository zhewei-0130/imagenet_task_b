import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset.imagenet_mini import ImageNetMiniDataset
import argparse
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
args = parser.parse_args()

data_root = "./"
test_txt = "./data/test.txt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_set = ImageNetMiniDataset(data_root, test_txt, transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model = models.resnet34(weights=None, num_classes=100)
ckpt = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(ckpt["state_dict"])
model.eval()

total_correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        output = model(imgs)
        total_correct += (output.argmax(1) == labels).sum().item()

acc = total_correct / len(test_loader.dataset)
print(f"✅ Test Accuracy: {acc:.4f}")
os.makedirs("logs", exist_ok=True)
with open("logs/test_result_resnet34.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["model", "accuracy"])
    writer.writerow(["resnet34", acc])