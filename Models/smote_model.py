import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import pickle
import traceback
from imblearn.over_sampling import SMOTE  # For handling class imbalance

# Import functions from model.py
from model import train_model, evaluate_model, plot_roc_curve

def load_data():
    # Load the already prepared pickled datasets
    try:
        import os
        import pickle
        
        # Get paths to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'Data')
        
        # Paths to pickled train/test data (SMOTE version)
        x_train_path = os.path.join(data_dir, 'X_train_pickled_smote.pkl')
        x_test_path = os.path.join(data_dir, 'X_test_pickled_smote.pkl')
        y_train_path = os.path.join(data_dir, 'y_train_pickled_smote.pkl')
        y_test_path = os.path.join(data_dir, 'y_test_pickled_smote.pkl')
        
        # Load the data directly
        X_train = pd.read_pickle(x_train_path)
        X_test = pd.read_pickle(x_test_path)
        y_train = pd.read_pickle(y_train_path)
        y_test = pd.read_pickle(y_test_path)
        
        print(f"SMOTE data loaded successfully")
        print(f"Training set shape: {X_train.shape}, {y_train.shape}")
        print(f"Test set shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

# Note: apply_smote function removed as we're loading pre-processed SMOTE data directly

# Functions train_model, evaluate_model, and plot_roc_curve are imported from model.py

def main():
    try:
        # Load data - returns train/test splits with SMOTE already applied
        X_train, X_test, y_train, y_test = load_data()
        if X_train is None:
            return
        
        print("Starting SMOTE model training...")
        # Train model (using function imported from model.py)
        try:
            model = train_model(X_train, y_train)
            print("SMOTE model training successful")
            
            # Save the model using pickle
            try:              
                # Create models directory if it doesn't exist
                models_dir = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(models_dir, exist_ok=True)
                
                # Save the model to a pickle file
                model_path = os.path.join(models_dir, 'logistic_regression_tuned_smote_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"SMOTE model successfully saved to {model_path}")
            except Exception as e:
                print(f"Error saving SMOTE model: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"Error during SMOTE model training: {e}")
            traceback.print_exc()
            return
        
        print("Starting SMOTE model evaluation...")
        # Evaluate model (using function imported from model.py)
        try:
            accuracy, report, cm = evaluate_model(model, X_test, y_test)
            print("SMOTE model evaluation successful")
        except Exception as e:
            print(f"Error during SMOTE model evaluation: {e}")
            traceback.print_exc()
            return
        
        print("Starting ROC curve plotting...")
        # Plot ROC Curve (using function imported from model.py)
        try:
            # Setting interactive mode for matplotlib to prevent blocking
            plt.ion()
            roc_auc = plot_roc_curve(model, X_test, y_test)
            print(f"SMOTE-enhanced model training and evaluation completed. AUC: {roc_auc:.4f}")
            
            # Save the ROC curve figure
            figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, 'smote_roc_curve.png'))
            print(f"ROC curve saved to {os.path.join(figures_dir, 'smote_roc_curve.png')}")
        except Exception as e:
            print(f"Error during ROC curve plotting: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
