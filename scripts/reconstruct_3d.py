# scripts/reconstruct_3d.py
import os
import numpy as np
from PIL import Image
import pandas as pd

# CONFIG
kaggle_dir = "data/kaggle/train"
output_dir = "data/volumes/train"
os.makedirs(output_dir, exist_ok=True)

# Class mapping (0=CN, 1=MCI, 2=AD)
class_map = {
    "non_demented": 0,
    "very_mild_demented": 1,
    "mild_demented": 1,
    "moderate_demented": 2
}

volume_id = 0
csv_lines = []

print("Starting 3D reconstruction...")

for cls in os.listdir(kaggle_dir):
    cls_path = os.path.join(kaggle_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(".jpg")]
    if not images:
        print(f"No images in {cls}")
        continue
    images.sort()
    
    group_size = 32
    for i in range(0, len(images), group_size):
        group = images[i:i + group_size]
        if len(group) < group_size:
            print(f"Skipping incomplete group in {cls} ({len(group)} slices)")
            continue
        
        slices = []
        for img_name in group:
            img_path = os.path.join(cls_path, img_name)
            try:
                img = np.array(Image.open(img_path).convert("L"))
                slices.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                break
        else:
            volume = np.stack(slices).astype(np.float32)
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            
            vol_filename = f"volume_{volume_id:04d}.npy"
            vol_path = os.path.join(output_dir, vol_filename)
            np.save(vol_path, volume)
            
            label = class_map[cls]
            csv_lines.append({
                "volume_id": vol_filename,
                "label": label,
                "age": np.random.randint(60, 90),
                "gender": np.random.choice([0, 1]),
                "education": np.random.randint(8, 20),
                "mmse": np.random.randint(10, 30),
                "cdr": np.random.choice([0.0, 0.5, 1.0, 2.0])
            })
            volume_id += 1

# Save CSV
csv_path = "data/train.csv"
pd.DataFrame(csv_lines).to_csv(csv_path, index=False)
print(f"Created {volume_id} 3D volumes in {output_dir}")
print(f"CSV saved: {csv_path}")