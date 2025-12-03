# import torch
# from src.models import DementiaResNet

# MODEL_PATH = "models/best_model.pth"  # adjust path if needed

# print("🔍 Checking model file...")

# # Choose device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize the model
# model = DementiaResNet(num_classes=4)  # ✅ parenthesis fixed
# model.to(device)

# # Load the checkpoint
# checkpoint = torch.load(MODEL_PATH, map_location=device)

# # Load weights safely
# try:
#     model.load_state_dict(checkpoint, strict=False)
#     print("✅ Model weights loaded successfully (with flexible matching).")
# except Exception as e:
#     print(f"⚠️ Error loading model weights: {e}")

# # Print model summary
# print("\n🧠 Model Architecture:\n")
# print(model)

# # Test forward pass with dummy input
# dummy_input = torch.randn(1, 3, 224, 224).to(device)
# with torch.no_grad():
#     output = model(dummy_input)
# print("\n✅ Forward pass successful. Output shape:", output.shape)


# src/check_model.py
import torch
import torch.nn.functional as F
import os
import sys

# adjust these imports to match your repository structure
from src.models import build_model  # if your function name differs, replace accordingly
from src.dataset import QuickDataset  # placeholder: replace with the dataset class name that returns (img, label, mask)

MODEL_PATH = "models/best_model.pth"
SAMPLE_INDEX = 0

def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model()  # replace if your model builder signature is different
    # handle both full state dict or wrapped dict
    try:
        model.load_state_dict(ckpt.get("state_dict", ckpt))
    except Exception as e:
        print("State dict load failed:", e)
        # try name conversion for DataParallel saved models
        sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(sd)
    model.eval()
    return model

def run_quick_check():
    if not os.path.exists(MODEL_PATH):
        print("Model not found at", MODEL_PATH)
        return
    model = load_model(MODEL_PATH)
    print("Model loaded. Sending a random sample through the model...")

    # Load sample from your dataset - adjust to your actual Dataset class and path
    try:
        ds = QuickDataset("data/preprocessed")  # update folder path if different
        img, label = ds[SAMPLE_INDEX]  # expects tensor shape [C,H,W] or [C,D,H,W]
    except Exception as e:
        print("Dataset load error - edit src/check_model.py to use your dataset class. Error:", e)
        return

    x = img.unsqueeze(0)  # add batch dim
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()
    print("Sample true label:", label)
    print("Output logits shape:", out.shape)
    print("Output probabilities:", probs)

if __name__ == "__main__":
    run_quick_check()
