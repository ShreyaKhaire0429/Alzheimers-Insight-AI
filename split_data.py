import os
import shutil
import random

# Path to your main folder
DATA_DIR = "data"

# Train/Val/Test split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Output directories
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

# Create directories if they don't exist
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all class folders inside data/
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
           and d not in ["train", "val", "test"]]

print(f"Detected classes: {classes}")

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    print(f"\n📁 {cls}: {n_total} total → {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    # Create subfolders for each class
    for split_name, split_imgs in zip(
        ["train", "val", "test"], [train_imgs, val_imgs, test_imgs]
    ):
        split_folder = os.path.join(DATA_DIR, split_name, cls)
        os.makedirs(split_folder, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_folder, img)
            shutil.copy(src, dst)  # Copy instead of move to preserve originals

print("\n✅ Data successfully split into train/val/test!")
