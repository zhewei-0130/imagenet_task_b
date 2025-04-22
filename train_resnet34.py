# train_resnet34.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataset.imagenet_mini import ImageNetMiniDataset
import os
import csv
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data_root = "./"
    train_txt = "./data/train.txt"
    val_txt = "./data/val.txt"

    train_set = ImageNetMiniDataset(data_root, train_txt, transform)
    val_set = ImageNetMiniDataset(data_root, val_txt, transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet34(weights=None, num_classes=100).to(device)

    total_params = count_parameters(model)
    print(f"📦 模型參數總數: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/train_log_resnet34.csv"
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    for epoch in range(10):
        print(f"\n📦 Epoch {epoch+1}/10")
        model.train()
        total_loss, total_correct = 0, 0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (output.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_loader.dataset)
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"✅ Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss, val_correct = 0, 0
        val_bar = tqdm(val_loader, desc="Evaluating")
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = criterion(output, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (output.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"✅ Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, train_acc, val_acc])

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, f"logs/resnet34_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()