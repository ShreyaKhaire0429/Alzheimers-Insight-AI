# train_multimodal.py (repo root)
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from src.models import build_multimodal_model, save_model

# Dummy dataset example: replace with your real dataset
class SimpleMultimodalDataset(Dataset):
    def __init__(self, image_tensors, clinical_array, labels):
        self.images = image_tensors        # list or tensor [N,3,H,W]
        self.clinical = clinical_array     # numpy [N, clinical_dim]
        self.labels = labels               # numpy [N]

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.clinical[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# load your training arrays here
# image_tensors = ... (torch.FloatTensor [N,3,H,W])
# clinical_array = ... (np.array [N,clinical_dim])
# labels = ... (np.array [N])

# Example placeholders - replace:
# image_tensors, clinical_array, labels = load_your_data()

# dataset = SimpleMultimodalDataset(image_tensors, clinical_array, labels)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_multimodal_model(clinical_dim=2, num_classes=3).to(device)

# Optionally freeze CNN backbone:
for p in model.cnn.parameters():
    p.requires_grad = True  # set False to freeze

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, clin, labels in loader:
        imgs = imgs.to(device)
        clin = clin.to(device)
        labels = labels.to(device)
        optim.zero_grad()
        outputs = model(imgs, clin)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

# save
save_model(model, "models/multimodal_best.pth")
print("Saved multimodal model.")


from src.models import build_multimodal_model, save_model
m = build_multimodal_model(clinical_dim=2, num_classes=3)
save_model(m, "models/multimodal_best.pth")
