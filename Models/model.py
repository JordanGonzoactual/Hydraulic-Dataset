import pandas as pd
import matplotlib.pyplot as plt
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
    # Load CSV files
    try:
        profile_data = pd.read_csv('../Data/profile.txt')
        labels_data = pd.read_csv('../Data/profiles_labels.txt')
        print(f"Data loaded successfully: {profile_data.shape}, {labels_data.shape}")
        return profile_data, labels_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def preprocess_data(profile_data, labels_data):
    # Preprocessing steps
    # Extracting features and target variable
    X = profile_data.iloc[:, 1:]  # All feature columns
    y = labels_data.iloc[:, 1]    # Target variable (second column in labels data)
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Logistic Regression CV model
    model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=1)
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
    # Load data
    profile_data, labels_data = load_data()
    if profile_data is None or labels_data is None:
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(profile_data, labels_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    
    # Plot ROC Curve
    roc_auc = plot_roc_curve(model, X_test, y_test)
    
    print(f"Model training and evaluation completed. AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
