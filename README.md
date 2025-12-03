# 🧠 Multimodal Explainable Alzheimer’s Detection

### 📌 Overview
This project detects Alzheimer's stages from MRI brain images using a deep learning CNN model.  
It includes explainability (Grad-CAM) and a Streamlit app for user interaction.

### 🧱 Folder Structure
MINI_2/
│
├── Data/ → Contains MRI datasets (by class)
├── src/ → Source code for data processing & training
├── streamlit_app/ → Web app (Streamlit)
├── models/ → Saved model weights
├── results/ → Plots, GradCAM images, metrics
├── notebooks/ → Jupyter visualizations
└── requirements.txt → Python dependencies

---

### 🚀 Steps to Run
1. Create environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

Train model:
python src/train.py

Run Streamlit app:
streamlit run streamlit_app/app.py

---

### 3. Add a `.gitignore` file (optional but good)
Create file `.gitignore` and paste:


.venv/
pycache/
models/
results/
*.pyc
*.pkl
.DS_Store


This ensures unnecessary files don’t get pushed to Git or clutter your repo.

---

### 4. Verify You Have All Python Files

| File | Purpose |
|------|----------|
| `data_prep.py` | MRI preprocessing (resize, normalize, save npy) |
| `dataset.py` | Dataset loader for training |
| `models.py` | CNN or ResNet model |
| `train.py` | Training loop |
| `explain.py` | GradCAM explainability |
| `test_explainability.py` | Optional — test explain.py results |
| `app.py` | Streamlit UI for predictions |

If you have all of these (✅ yes, you do), you’re complete.

---

###📦 Dataset Used

OASIS Alzheimer’s MRI Dataset — Kaggle
Contains:
3,700+ MRI images (128×128)
Labels: CN, MCI, AD
Clinical information (Age, MMSE)

Dataset link:
https://www.kaggle.com/datasets/ebrahimelgazar/oasis-mri-dataset

---

Model Architecture Details:
MRI Image Model (CNN)
ResNet-18
Extracts spatial brain features
Clinical Model (MLP)

Input:
Age
MMSE score
Fusion Model
Merges MRI + Clinical feature vectors
Outputs CN / MCI / AD

---

Technologies Used:
Python · PyTorch · OpenCV
Streamlit · NumPy · Pandas
SHAP · Matplotlib · ReportLab
Scikit-learn
