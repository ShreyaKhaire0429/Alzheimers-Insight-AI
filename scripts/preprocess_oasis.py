import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom

# CONFIG
OASIS_DIR = "oasis"
OUTPUT_DIR = "data/volumes/processed"
CSV_OUT = "data/train.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create dummy clinical data if no CSV exists
clinical_csv = f"{OASIS_DIR}/clinical_data.csv"
if not os.path.exists(clinical_csv):
    print("clinical_data.csv not found → creating dummy metadata...")
    subjects = [f for f in os.listdir(OASIS_DIR) if f.startswith("OAS1_") and os.path.isdir(f"{OASIS_DIR}/{f}")]
    dummy_data = []
    for subj in subjects:
        subj_id = subj.split("_")[0]  
        sess = subj.split("_")[-1]    
        dummy_data.append({
            "Subject": subj_id,
            "Session": sess,
            "Age": np.random.randint(60, 90),
            "M/F": np.random.choice(["M", "F"]),
            "Educ": np.random.randint(8, 20),
            "MMSE": np.random.randint(15, 30),
            "CDR": np.random.choice([0.0, 0.5, 1.0, 2.0])
        })
    pd.DataFrame(dummy_data).to_csv(clinical_csv, index=False)
    print(f"Created dummy {clinical_csv}")

# Load clinical data
clinical_df = pd.read_csv(clinical_csv)
print(f"Loaded {len(clinical_df)} subjects")

data = []
vol_id = 0

for _, row in clinical_df.iterrows():
    subj_id = row["Subject"]
    sess = row["Session"]
    nii_path = f"{OASIS_DIR}/{subj_id}_{sess}/mpr-1.nifti.nii.gz"
    
    if not os.path.exists(nii_path):
        print(f"Missing: {nii_path}")
        continue
    
    # Load 3D volume
    img = nib.load(nii_path)
    vol = img.get_fdata()
    vol = np.transpose(vol, (2, 0, 1))  # (D, H, W)
    
    # Resize to (64, 128, 128)
    vol = zoom(vol, (64/vol.shape[0], 128/vol.shape[1], 128/vol.shape[2]), order=1)
    
    # Normalize
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    
    # Save
    vol_file = f"volume_{vol_id:04d}.npy"
    vol_path = f"{OUTPUT_DIR}/{vol_file}"
    np.save(vol_path, vol.astype(np.float32))
    
    # Label from CDR
    cdr = row["CDR"]
    label = 0 if cdr == 0 else (1 if cdr <= 0.5 else 2)
    
    data.append({
        "volume_id": vol_file,
        "label": label,
        "age": row["Age"],
        "gender": 1 if row["M/F"] == "M" else 0,
        "education": row.get("Educ", 12),
        "mmse": row["MMSE"],
        "cdr": cdr
    })
    vol_id += 1

# Save CSV
pd.DataFrame(data).to_csv(CSV_OUT, index=False)
print(f"Preprocessed {vol_id} volumes → {OUTPUT_DIR}")
print(f"CSV saved → {CSV_OUT}")