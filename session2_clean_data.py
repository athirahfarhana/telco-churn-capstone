"""
Data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def clean_data(df):
    """Clean the raw data."""
    df = df.copy()
    
    # Convert TotalCharges to numeric (handle blanks)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # For empty TotalCharges, fill with 0 (new customers with no charges)
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert Churn to binary (Yes=1, No=0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

# Load and clean
from src.data_loader import load_data

df = load_data("data/raw/Telco-Customer-Churn.csv")
df_clean = clean_data(df)

# Save cleaned data
df_clean.to_csv("data/processed/telco_cleaned.csv", index=False)

print("Data cleaned and saved to data/processed/telco_cleaned.csv")
print(f"Shape: {df_clean.shape}")
print(f"Churn values: {df_clean['Churn'].unique()}")
print(f"TotalCharges (first 10): {df_clean['TotalCharges'].head(10).tolist()}")
