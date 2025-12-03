import os
import numpy as np

input_dir = "data/volumes/train"
output_dir = "data/volumes/processed"
os.makedirs(output_dir, exist_ok=True)

target_shape = (64, 128, 128)  # D, H, W

print("Preprocessing 3D volumes...")
for f in os.listdir(input_dir):
    if not f.endswith(".npy"): continue
    vol = np.load(os.path.join(input_dir, f))
    
    # Resize if needed
    if vol.shape != target_shape:
        import scipy.ndimage as ndimage
        vol = ndimage.zoom(vol, (
            target_shape[0]/vol.shape[0],
            target_shape[1]/vol.shape[1],
            target_shape[2]/vol.shape[2]
        ), order=1)
    
    # Normalize
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    
    np.save(os.path.join(output_dir, f), vol.astype(np.float32))

print(f"Processed volumes saved → {output_dir}")