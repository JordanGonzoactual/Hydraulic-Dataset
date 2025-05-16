import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

def load_data():
    # Load the already prepared pickled datasets
    try:
        import os
        import pickle
        
        # Get paths to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'Data')
        
        # Paths to pickled train/test data
        x_train_path = os.path.join(data_dir, 'X_train_pickled.pkl')
        x_test_path = os.path.join(data_dir, 'X_test_pickled.pkl')
        y_train_path = os.path.join(data_dir, 'y_train_pickled.pkl')
        y_test_path = os.path.join(data_dir, 'y_test_pickled.pkl')
        
        # Load the data directly
        X_train = pd.read_pickle(x_train_path)
        X_test = pd.read_pickle(x_test_path)
        y_train = pd.read_pickle(y_train_path)
        y_test = pd.read_pickle(y_test_path)
        
        print(f"Data loaded successfully")
        print(f"Training set shape: {X_train.shape}, {y_train.shape}")
        print(f"Test set shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    # Logistic Regression CV model
    print("Initializing LogisticRegressionCV model...")
    model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=1)
    
    # Initialize progress bar for training
    print("Starting model training with progress tracking...")
    with tqdm(total=100, desc='Training model', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        # Create a custom fit method that updates the progress bar
        orig_fit = model.fit
        
        # Define a wrapper to update progress during training
        def fit_with_progress(*args, **kwargs):
            result = orig_fit(*args, **kwargs)
            # Update progress bar to show completion
            pbar.update(100)
            return result
        
        # Replace the original fit method with our progress-tracking version
        model.fit = fit_with_progress
        
        # Start training
        model.fit(X_train, y_train)
        
    print("Model trained successfully")
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()
    
    return accuracy, report, cm

def plot_roc_curve(model, X_test, y_test):
    # ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return roc_auc

def main():
    try:
        # Load data - now directly returns train/test splits
        X_train, X_test, y_train, y_test = load_data()
        if X_train is None:
            return
        
        print("Starting model training...")
        # Train model
        try:
            model = train_model(X_train, y_train)
            print("Model training successful")
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("Starting model evaluation...")
        # Evaluate model
        try:
            accuracy, report, cm = evaluate_model(model, X_test, y_test)
            print("Model evaluation successful")
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("Starting ROC curve plotting...")
        # Plot ROC Curve - making this non-blocking
        try:
            # Setting interactive mode for matplotlib to prevent blocking
            plt.ion()
            roc_auc = plot_roc_curve(model, X_test, y_test)
            # Save figure instead of showing it interactively
            plt.savefig('roc_curve.png')
            plt.close()
            print(f"ROC curve saved to roc_curve.png")
        except Exception as e:
            print(f"Error during ROC curve plotting: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"Model training and evaluation completed. AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"Unexpected error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
