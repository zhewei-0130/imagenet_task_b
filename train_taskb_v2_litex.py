import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.imagenet_mini import ImageNetMiniDataset
from models.SimpleNetV2_LiteX import SimpleNetV2_LiteX
from tqdm import tqdm
import os
import csv

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]), ])

    data_root = "./"
    train_txt = "./data/train.txt"
    val_txt = "./data/val.txt"

    train_set = ImageNetMiniDataset(data_root, train_txt, transform)
    val_set = ImageNetMiniDataset(data_root, val_txt, val_transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNetV2_LiteX(num_classes=100).to(device)
    print(f"📦 模型參數總數: {count_params(model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/train_log_taskb_v2_litex.csv"
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(20):
        model.train()
        total_loss, correct = 0, 0
        loop = tqdm(train_loader, desc=f"[LiteX] Epoch {epoch+1}/20")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"✅ Epoch {epoch+1}｜Train Acc: {train_acc:.4f}｜Val Acc: {val_acc:.4f}")
        torch.save({"epoch": epoch+1, "state_dict": model.state_dict()}, f"logs/taskb_v2_litex_epoch{epoch+1}.pth")

        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])