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

💡 Future Scope (Major Project)

Integrate clinical data features.

Add SHAP-based explainability.

Build a more interactive dashboard.


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

### 5. (Optional) Add a `config.json` File  
For smoother training — store constants like data paths, epochs, batch size here:

`config.json`
```json
{
  "data_path": "Data/",
  "model_path": "models/best_model.pth",
  "img_size": 128,
  "batch_size": 8,
  "epochs": 15,
  "learning_rate": 0.0001
}