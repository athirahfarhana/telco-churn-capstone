"""
Data preprocessing and cleaning module.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    """
    Clean the raw data.
    - Convert TotalCharges to numeric
    - Convert Churn to binary
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric (handle blanks)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert Churn to binary (Yes=1, No=0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

def prepare_features(df):
    """
    Prepare features for ML model.
    - Remove customerID
    - Encode categorical variables
    - Scale numeric features
    """
    df = df.copy()
    
    # Remove NaN values
    df = df.dropna()
    
    # Remove customerID (not a feature)
    df = df.drop(columns=['customerID'])
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y

def validate_features(X, y):
    """Validate that X and y have no NaN values."""
    if X.isna().sum().sum() > 0:
        raise ValueError("X contains NaN values")
    if y.isna().sum() > 0:
        raise ValueError("y contains NaN values")
    return True
