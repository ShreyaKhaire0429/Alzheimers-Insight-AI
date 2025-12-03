from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import sys
from pathlib import Path

# Make D:\mini_2 visible to Python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# Load best model
# Predict on test set
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")