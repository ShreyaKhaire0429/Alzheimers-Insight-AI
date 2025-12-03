import os
import pandas as pd
import numpy as np

base_dir = "data"
data = []

# Exact MMSE Ranges (Medical Standard)
MMSE_RANGES = {
    'CN': (27, 31),   # 27–30
    'MCI': (21, 27),  # 21–26
    'AD': (0, 21)     # 0–20
}

class_map = {
    'Non Demented': 'CN',
    'Very Mild Dementia': 'MCI',
    'Mild Dementia': 'MCI',
    'Moderate Dementia': 'AD'
}

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path): continue
    label = class_map.get(folder)
    if not label: continue

    low, high = MMSE_RANGES[label]
    for img in os.listdir(folder_path):
        if img.lower().endswith(('.jpg', '.png', '.jpeg')):
            mmse = np.random.randint(low, high)
            cdr = 0.0 if label == 'CN' else (0.5 if label == 'MCI' else np.random.choice([1.0, 2.0, 3.0]))

            data.append({
                "image": os.path.join(folder, img),
                "label": label,
                "age": np.random.randint(70, 91),
                "mmse": mmse,
                "gender": np.random.choice([0, 1]),
                "education": np.random.randint(8, 20),
                "cdr": cdr
            })

df = pd.DataFrame(data)
df.to_csv("data/train.csv", index=False)
print(f"Created {len(df)} samples")
print("\nMMSE Distribution:")
print(df.groupby('label')['mmse'].agg(['min', 'max', 'mean']).round(1))