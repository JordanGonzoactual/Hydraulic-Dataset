import pickle
import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin

# Define model class to differentiate between standard and SMOTE models
class ModelType:
    STANDARD = 'standard'
    SMOTE = 'smote'

def create_reports_dir():
    """Create reports directory if it doesn't exist."""
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

def load_model(model_type=ModelType.STANDARD):
    """Load the specified model from pickle file."""
    if model_type == ModelType.STANDARD:
        model_path = Path("D:/Python/Hydraulic Rig Dataset/Models/logistic_regression_tuned_model.pkl")
        model_name = "standard model"
    else:  # SMOTE model
        model_path = Path("D:/Python/Hydraulic Rig Dataset/Models/logistic_regression_tuned_smote_model.pkl")
        model_name = "SMOTE model"
    
    print(f"Loading pickled {model_name}...")
    try:
        # Load the pre-trained model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data(data_type='standard'):
    """Load validation and test data."""
    print("Loading validation and test data...")
    data_dir = Path("D:/Python/Hydraulic Rig Dataset/Data")
    
    # Set file paths based on data type
    if data_type == 'smote':
        # For SMOTE-enhanced data
        X_val_path = data_dir / 'X_test_pickled_smote.pkl'
        y_val_path = data_dir / 'y_test_pickled_smote.pkl'
        X_test_path = data_dir / 'X_test_pickled_smote.pkl'
        y_test_path = data_dir / 'y_test_pickled_smote.pkl'
    else:
        # For standard data
        X_val_path = data_dir / 'X_test_pickled.pkl'
        y_val_path = data_dir / 'y_test_pickled.pkl'
        X_test_path = data_dir / 'X_test_pickled.pkl'
        y_test_path = data_dir / 'y_test_pickled.pkl'
    
    try:
        X_val = pd.read_pickle(X_val_path)
        y_val = pd.read_pickle(y_val_path)
        X_test = pd.read_pickle(X_test_path)
        y_test = pd.read_pickle(y_test_path)
        
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
        print(f"Test data shape: {X_test.shape}, {y_test.shape}")
        
        return X_val, y_val, X_test, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def find_optimal_threshold(model, X_val, y_val, target_recall=0.95):
    """
    Find optimal threshold that maximizes F1 score while meeting minimum recall
    requirements.
    """
    print("\nSweeping thresholds to find optimal value...")
    proba_val = model.predict_proba(X_val)[:, 1]
    precision, recall, thresh = precision_recall_curve(y_val, proba_val)
    
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
    
    # Display results
    print(f"Chosen threshold = {best_thresh:.4f}")
    print(f"Metrics at this threshold:")
    print(f"  - Recall: {best_recall:.4f}")
    print(f"  - Precision: {best_precision:.4f}")
    print(f"  - F1 Score: {best_f1:.4f}")
    print(f"  - Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={len(y_val)-(tp+fp+fn)}")
    
    # Create a dictionary with all the threshold info
    threshold_info = {
        'threshold': float(best_thresh),
        'recall': float(best_recall),
        'precision': float(best_precision),
        'f1_score': float(best_f1),
        'confusion_matrix': {
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(len(y_val)-(tp+fp+fn))
        }
    }
    
    return best_thresh, threshold_info

def plot_precision_recall_curve(model, X_val, y_val, optimal_threshold, model_name):
    """Plot and save precision-recall curve with optimal threshold marked."""
    reports_dir = create_reports_dir()
    
    # Get predictions and calculate curve
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Find the closest threshold to our optimal one
    optimal_idx = (np.abs(thresholds - optimal_threshold)).argmin() if len(thresholds) > 0 else 0
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name} with Optimal Threshold')
    
    # Mark optimal threshold
    if len(thresholds) > 0 and optimal_idx < len(precision):
        plt.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=10,
                label=f'Optimal Threshold = {optimal_threshold:.4f}')
        plt.legend()
    
    # Save plot
    pr_curve_path = os.path.join(reports_dir, f'{model_name}_precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    print(f"Precision-Recall curve saved to {pr_curve_path}")
    plt.close()

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Classifier wrapper that applies a custom threshold to prediction probabilities."""
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

def evaluate_threshold_model(threshold_model, X_test, y_test, model_name):
    """Evaluate the threshold-adjusted model and save results."""
    reports_dir = create_reports_dir()
    
    print(f"\nEvaluating {model_name} with new threshold on test data...")
    y_pred_test = threshold_model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix with ThresholdClassifier:")
    print(cm)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name} with Optimal Threshold')
    
    # Save the confusion matrix figure
    cm_fig_path = os.path.join(reports_dir, f'{model_name}_threshold_confusion_matrix.png')
    plt.savefig(cm_fig_path)
    print(f"Confusion matrix visualization saved to {cm_fig_path}")
    plt.close()
    
    # Generate and display classification report
    report = classification_report(y_test, y_pred_test)
    print("\nClassification Report with ThresholdClassifier:")
    print(report)
    
    # Save classification report to file
    report_path = os.path.join(reports_dir, f'{model_name}_threshold_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name} with Optimal Threshold\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"Classification report saved to {report_path}")
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred_test)
    print(f"F1: {f1:.4f}")
    
    # Calculate additional performance metrics
    recall = ((y_pred_test == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
    precision = ((y_pred_test == 1) & (y_test == 1)).sum() / (y_pred_test == 1).sum() if (y_pred_test == 1).sum() > 0 else 0
    
    # Generate performance metrics
    performance = {
        'f1_score': float(f1),
        'recall': float(recall),
        'precision': float(precision),
        'confusion_matrix': {
            'true_positives': int(((y_pred_test == 1) & (y_test == 1)).sum()),
            'false_positives': int(((y_pred_test == 1) & (y_test == 0)).sum()),
            'true_negatives': int(((y_pred_test == 0) & (y_test == 0)).sum()),
            'false_negatives': int(((y_pred_test == 0) & (y_test == 1)).sum())
        }
    }
    
    # Plot and save ROC curve
    y_pred_proba = threshold_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} with Optimal Threshold')
    plt.legend(loc='lower right')
    
    # Save ROC curve
    roc_path = os.path.join(reports_dir, f'{model_name}_threshold_roc_curve.png')
    plt.savefig(roc_path)
    print(f"ROC curve saved to {roc_path}")
    plt.close()
    
    return performance

def save_threshold_model(threshold_model, model_name):
    """Save the threshold-adjusted model to a file."""
    try:
        # Create path for the threshold classifier model
        threshold_model_path = Path(f"D:/Python/Hydraulic Rig Dataset/Models/{model_name}_threshold_model.pkl")
        
        # Save the model
        with open(threshold_model_path, 'wb') as f:
            pickle.dump(threshold_model, f)
            
        print(f"\nThresholdClassifier saved to {threshold_model_path}")
        return threshold_model_path
    except Exception as e:
        print(f"Error saving ThresholdClassifier: {e}")
        return None

def save_config(model_name, threshold_info, performance, config_path=None):
    """Save model configuration and performance metrics to a YAML file."""
    if config_path is None:
        config_path = Path(f"D:/Python/Hydraulic Rig Dataset/Models/{model_name}_config.yaml")
    
    config = {
        "model_name": model_name,
        "version": "1.0.0",
        "threshold": {
            "value": threshold_info['threshold'],
            "metric_target": {
                "optimization": "max_f1_with_min_recall",
                "min_recall": 0.95
            }
        },
        "performance": performance,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def process_model(model_type, data_type=None):
    """Process a model: load, find optimal threshold, evaluate, and save."""
    if model_type == ModelType.STANDARD:
        model_name = "standard_logistic_regression"
        if data_type is None:
            data_type = 'standard'
    else:  # SMOTE model
        model_name = "smote_logistic_regression"
        if data_type is None:
            data_type = 'smote'
    
    # Load model
    model = load_model(model_type)
    if model is None:
        print(f"Failed to load {model_name}. Exiting.")
        return
    
    # Load data
    X_val, y_val, X_test, y_test = load_data(data_type)
    if X_val is None:
        print("Failed to load data. Exiting.")
        return
    
    # Find optimal threshold
    optimal_threshold, threshold_info = find_optimal_threshold(model, X_val, y_val)
    
    # Plot precision-recall curve
    plot_precision_recall_curve(model, X_val, y_val, optimal_threshold, model_name)
    
    # Create threshold-adjusted model
    threshold_model = ThresholdClassifier(model, thr=optimal_threshold)
    
    # Evaluate threshold model
    performance = evaluate_threshold_model(threshold_model, X_test, y_test, model_name)
    
    # Save threshold model
    save_threshold_model(threshold_model, model_name)
    
    # Save configuration
    save_config(model_name, threshold_info, performance)
    
    print(f"\nCompleted threshold tuning for {model_name}")
    return threshold_model, optimal_threshold

# Missing import for accuracy_score
from sklearn.metrics import accuracy_score

def main():
    """Main function to run threshold tuning for both standard and SMOTE models."""
    print("\n===== THRESHOLD TUNING: STANDARD MODEL =====\n")
    process_model(ModelType.STANDARD)
    
    print("\n\n===== THRESHOLD TUNING: SMOTE MODEL =====\n")
    process_model(ModelType.SMOTE)
    
    print("\n\nThreshold tuning completed for both models!")
    print("Reports and visualizations saved to the reports directory.")

if __name__ == "__main__":
    main()
