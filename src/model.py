import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_and_preprocess(filepath):
    """
    Load and preprocess the Student Performance dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Encode categorical variable 'Extracurricular Activities'
    # Yes -> 1, No -> 0
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    
    # Selecting target and features
    target = 'Performance Index'
    features = [
        'Hours Studied', 
        'Previous Scores', 
        'Extracurricular Activities', 
        'Sleep Hours', 
        'Sample Question Papers Practiced'
    ]
    
    X = df[features]
    y = df[target]
    
    return X, y, features

def train_model(X, y):
    """
    Train a Linear Regression model and calculate metrics.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    metrics = {
        'R-squared': r2,
        'Adjusted R-squared': adj_r2,
        'RMSE': rmse,
        'MAE': mae
    }
    
    return model, metrics

if __name__ == "__main__":
    # Path to dataset relative to this script or current working directory
    data_path = "./dataset/Student_Performance.csv"
    
    print(f"Loading data from {data_path}...")
    X, y, features = load_and_preprocess(data_path)
    
    print("Training model...")
    model, metrics = train_model(X, y)
    
    print("\n--- Model Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save model
    model_dir = "./applied-ml/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "student_performance_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
