# src/kfold.py
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Make D:\mini_2 visible to Python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
df = pd.read_csv('data/combined.csv')
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accs = []
for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_df.to_csv(f'data/fold_{fold}_train.csv', index=False)
    val_df.to_csv(f'data/fold_{fold}_val.csv', index=False)
    
    # Run training (modify train.py to accept fold)
    print(f"Training Fold {fold+1}")
    os.system(f"python src/train.py --fold {fold}")
    
    # Load best acc from log
    accs.append(best_acc_from_log)

print(f"CV Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}")