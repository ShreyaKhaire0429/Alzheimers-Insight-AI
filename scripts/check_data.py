# scripts/check_data.py
import os
from PIL import Image
import numpy as np

sample_dir = "data/preprocessed"  # update if different
files = []
for root, dirs, filenames in os.walk(sample_dir):
    for f in filenames:
        if f.lower().endswith((".png",".jpg",".nii",".nii.gz")):
            files.append(os.path.join(root,f))
print("Found", len(files), "files under", sample_dir)
# show first few sample shapes (for png/jpg). For NIfTI we print header info.
for f in files[:10]:
    print(f)
    if f.endswith((".png",".jpg")):
        im = Image.open(f)
        print("  size:", im.size, "mode:", im.mode)
    else:
        import nibabel as nib
        img = nib.load(f)
        print("  nifti shape:", img.shape, "affine diag:", np.diag(img.affine)[:3])


# scripts/check_data.py
import os
from PIL import Image
import numpy as np

kaggle_dir = "data/kaggle/train"

print(f"Scanning: {kaggle_dir}")
total = 0
for cls in os.listdir(kaggle_dir):
    cls_path = os.path.join(kaggle_dir, cls)
    if not os.path.isdir(cls_path): continue
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(".jpg")]
    print(f"{cls}: {len(imgs)} images")
    total += len(imgs)
    if len(imgs) > 0:
        sample = os.path.join(cls_path, imgs[0])
        img = Image.open(sample)
        print(f"  Sample: {imgs[0]} → size: {img.size}, mode: {img.mode}")
print(f"TOTAL: {total} images")