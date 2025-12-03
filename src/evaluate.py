import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import os

from src.models import DementiaResNet  # import your model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"
DATA_DIR = "data/val"
BATCH_SIZE = 32


def load_model():
    print("✅ Loading trained model...")
    model = DementiaResNet(num_classes=4)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    #  FIX: rename keys if they don’t start with "base_model."
    if not list(checkpoint.keys())[0].startswith("base_model."):
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_state_dict["base_model." + k] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()
    print(" Model loaded successfully.\n")
    return model


def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f" Loaded {len(dataset)} images across {len(dataset.classes)} classes.\n")
    return dataloader, dataset.classes


def evaluate(model, dataloader, class_names):
    print(" Evaluating model...\n")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Evaluation complete!\n")
    print("🔹 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n🔹 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    print("Evaluation script started...")
    model = load_model()
    dataloader, class_names = load_data()
    evaluate(model, dataloader, class_names)


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader):
    y_true, y_pred = [], []
    model.eval()
    for imgs, labels in dataloader:
        preds = model(imgs)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.argmax(dim=1).cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True)
    plt.show()

