import os
from src.explain import load_model, run_cam_on_slice

print("✅ test_explainability.py started running")

# Ensure results directories exist
os.makedirs("results/gradcam/compare", exist_ok=True)

print("🔹 Loading model...")
model = load_model()
print("✅ Model loaded successfully")

# Example test images (replace with your actual paths)
test_images = [
    "data/val/class1/img1.jpg",
    "data/val/class2/img2.jpg"
]

for img_path in test_images:
    if os.path.exists(img_path):
        out_name = os.path.basename(img_path).replace(".jpg", "_gradcam.png")
        out_path = os.path.join("results/gradcam/compare", out_name)
        pred, prob = run_cam_on_slice(model, img_path, out_path)
        print(f"✅ Saved Grad-CAM for {img_path} → {out_path}")
    else:
        print(f"⚠️ Skipping missing image: {img_path}")

print("🎯 Grad-CAM explainability test completed!")
