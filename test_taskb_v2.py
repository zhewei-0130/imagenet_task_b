import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.imagenet_mini import ImageNetMiniDataset
from models.SimpleNetV2 import SimpleNetV2
import argparse
import os
import csv
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_set = ImageNetMiniDataset("./", "data/test.txt", transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleNetV2(num_classes=100).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="[TaskBv2] Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(test_loader.dataset)
    print(f"✅ Test Accuracy: {acc:.4f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/test_result_taskb_v2.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy"])
        writer.writerow(["SimpleNetV2", acc])
