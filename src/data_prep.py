import os, shutil, random
from sklearn.model_selection import train_test_split
from PIL import Image

DATA_DIR = "Data"
OUTPUT_DIR = "Data_split"
IMG_SIZE = (128, 128)
TEST_RATIO = 0.2

def preprocess_images():
    classes = os.listdir(DATA_DIR)
    for c in classes:
        imgs = os.listdir(os.path.join(DATA_DIR, c))
        train_imgs, test_imgs = train_test_split(imgs, test_size=TEST_RATIO, random_state=42)

        for split, files in [("train", train_imgs), ("test", test_imgs)]:
            split_dir = os.path.join(OUTPUT_DIR, split, c)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                src = os.path.join(DATA_DIR, c, f)
                dst = os.path.join(split_dir, f)
                img = Image.open(src).convert("RGB").resize(IMG_SIZE)
                img.save(dst)

if __name__ == "__main__":
    preprocess_images()
    print("✅ Data preprocessing complete!")
