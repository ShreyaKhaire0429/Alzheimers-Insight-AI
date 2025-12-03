import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import MultimodalDataset
from src.models import Multimodal2DModel   # ← fixed import

# 1. Transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 2. Dataset
CSV_PATH = "data/train.csv"
ROOT_DIR = "data"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("Run: python scripts/create_csv.py first!")

dataset = MultimodalDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

BATCH_SIZE = 16
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_set)} | Val: {len(val_set)}")

# 3. Class weights
labels = [dataset[i][2] for i in range(len(dataset))]
class_counts = np.bincount(labels, minlength=3)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# 4. Model
model = Multimodal2DModel(clinical_dim=5, num_classes=3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


# 5. Training

EPOCHS = 30
PATIENCE = 7
best_acc = 0.0
patience_counter = 0
train_losses, val_losses, val_accs = [], [], []

MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for img, clin, label in train_loader:
        img, clin, label = img.to(device), clin.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(img, clin)
        loss = criterion(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = correct = total = 0
    with torch.no_grad():
        for img, clin, label in val_loader:
            img, clin, label = img.to(device), clin.to(device), label.to(device)
            out = model(img, clin)
            val_loss += criterion(out, label).item()
            pred = out.argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    acc = 100 * correct / total
    val_loss_avg = val_loss / len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss_avg)
    val_accs.append(acc)

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {acc:.2f}%")
    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "multimodal_2d_best.pth"))
        patience_counter = 0
        print(f" → BEST MODEL SAVED: {best_acc:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping!")
            break

# 6. Plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val"); plt.title("Loss"); plt.legend(); plt.grid()
plt.subplot(1,2,2); plt.plot(val_accs, label="Val Acc", color='green'); plt.title("Accuracy (%)"); plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
plt.close()

print(f"\nDONE! Best Accuracy: {best_acc:.2f}%")
print(f"Model: {os.path.join(MODEL_DIR, 'multimodal_2d_best.pth')}")