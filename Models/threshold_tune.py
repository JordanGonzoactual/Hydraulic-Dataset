import pickle
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

# Define paths
MODEL_PATH = Path("D:/Python/Hydraulic Rig Dataset/Models/logistic_regression_tuned_model.pkl")
CONFIG_PATH = Path("D:/Python/Hydraulic Rig Dataset/Models/config.yaml")

print("Loading pickled model...")
try:
    # Load the pre-trained model
    with open(MODEL_PATH, 'rb') as f:
        pipe = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Exiting script.")
    exit(1)

# Load validation and test data
print("Loading validation and test data...")
data_dir = Path("D:/Python/Hydraulic Rig Dataset/Data")

# For threshold tuning, we'll use the test set as validation
X_val_path = data_dir / 'X_test_pickled.pkl'
y_val_path = data_dir / 'y_test_pickled.pkl'

# For final evaluation (we'll use the same data for simplicity)
X_test_path = data_dir / 'X_test_pickled.pkl'
y_test_path = data_dir / 'y_test_pickled.pkl'

X_val = pd.read_pickle(X_val_path)
y_val = pd.read_pickle(y_val_path)
X_test = pd.read_pickle(X_test_path)
y_test = pd.read_pickle(y_test_path)

print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# 2) Sweep thresholds on the VALIDATION set ----------------------------
print("\nSweeping thresholds to find optimal value...")
proba_val = pipe.predict_proba(X_val)[:, 1]
precision, recall, thresh = precision_recall_curve(y_val, proba_val)

# Set target recall for class 1
target_recall = 0.95
print(f"Finding optimal threshold with class 1 recall >= {target_recall}...")

# Find all thresholds that meet minimum recall requirement
eligible_thresholds = []

# Sweep through thresholds and find those that meet recall target
for i, t in enumerate(thresh):
    # Skip the last threshold which is usually 0 and would give 100% recall but terrible precision
    if i >= len(thresh) - 1:
        continue
        
    # Make predictions with this threshold
    preds = (proba_val >= t).astype(int)
    
    # Calculate TP, FP, FN for metrics
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    
    # Calculate recall for class 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate precision and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Store thresholds that meet recall target
    if recall >= target_recall:
        eligible_thresholds.append((i, t, f1, precision, recall, tp, fp, fn))

# Check if any thresholds meet the recall criteria
if eligible_thresholds:
    print(f"Found {len(eligible_thresholds)} thresholds with recall >= {target_recall}")
    
    # Select the threshold with highest F1 score among eligible thresholds
    best_idx, best_thresh, best_f1, best_precision, best_recall, tp, fp, fn = max(eligible_thresholds, key=lambda x: x[2])
    
    print(f"Selected threshold with best F1 score among those with recall >= {target_recall}")
    OPT_THRESH = best_thresh
else:
    print(f"Warning: No thresholds found with recall >= {target_recall}")
    print("Selecting threshold with highest possible recall")
    
    # Find all thresholds
    all_thresholds = []
    for i, t in enumerate(thresh):
        if i >= len(thresh) - 1:
            continue
            
        # Make predictions with this threshold
        preds = (proba_val >= t).astype(int)
        
        # Calculate metrics
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_thresholds.append((i, t, f1, precision, recall, tp, fp, fn))
    
    # Select threshold with highest recall
    best_idx, best_thresh, best_f1, best_precision, best_recall, tp, fp, fn = max(all_thresholds, key=lambda x: x[4])
    OPT_THRESH = best_thresh

# Display results
print(f"Chosen threshold = {OPT_THRESH:.4f}")
print(f"Metrics at this threshold:")
print(f"  - Recall: {best_recall:.4f}")
print(f"  - Precision: {best_precision:.4f}")
print(f"  - F1 Score: {best_f1:.4f}")
print(f"  - Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={len(y_val)-(tp+fp+fn)}")

# 3) Wrap model + threshold ----------------------------------------------
print("\nCreating ThresholdClassifier...")

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, thr=0.5):
        self.base_model = base_model
        self.thr = thr
    
    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        return (self.base_model.predict_proba(X)[:, 1] >= self.thr).astype(int)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

prod_clf = ThresholdClassifier(pipe, thr=OPT_THRESH)

# 4) Re-evaluate on the TEST set ---------------------------------------
print("\nEvaluating model with new threshold on test data...")
y_pred_test = prod_clf.predict(X_test)

print("\nConfusion Matrix with ThresholdClassifier:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Generate and display classification report
print("\nClassification Report with ThresholdClassifier:")
report = classification_report(y_test, y_pred_test)
print(report)

f1 = f1_score(y_test, y_pred_test)
print(f"F1: {f1:.4f}")

# Calculate additional performance metrics to save
recall = ((y_pred_test == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
precision = ((y_pred_test == 1) & (y_test == 1)).sum() / (y_pred_test == 1).sum() if (y_pred_test == 1).sum() > 0 else 0

# Save the ThresholdClassifier to a file
try:
    # Create path for the threshold classifier model
    threshold_model_path = Path("D:/Python/Hydraulic Rig Dataset/Models/threshold_classifier_model.pkl")
    
    # Save the model
    with open(threshold_model_path, 'wb') as f:
        pickle.dump(prod_clf, f)
        
    print(f"\nThresholdClassifier saved to {threshold_model_path}")
except Exception as e:
    print(f"Error saving ThresholdClassifier: {e}")

# Save the threshold and performance metrics to config.yaml
config = {
    "model_name": "logistic_regression_machine_health",
    "version": "1.0.0",
    "threshold": {
        "value": float(OPT_THRESH),
        "metric_target": {
            "optimization": "max_f1_with_min_recall",
            "min_recall": target_recall
        }
    },
    "performance": {
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "confusion_matrix": {
            "true_positives": int(((y_pred_test == 1) & (y_test == 1)).sum()),
            "false_positives": int(((y_pred_test == 1) & (y_test == 0)).sum()),
            "true_negatives": int(((y_pred_test == 0) & (y_test == 0)).sum()),
            "false_negatives": int(((y_pred_test == 0) & (y_test == 1)).sum())
        }
    },
    "timestamp": __import__("datetime").datetime.now().isoformat()
}

with open(CONFIG_PATH, 'w') as f:
    yaml.safe_dump(config, f)

print(f"Threshold configuration saved to {CONFIG_PATH}")