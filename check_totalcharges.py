import pandas as pd
from src.data_loader import load_data

df = load_data("data/raw/Telco-Customer-Churn.csv")

print("TotalCharges column info:")
print(f"Data type: {df['TotalCharges'].dtype}")
print(f"Sample values: {df['TotalCharges'].head(20)}")
print(f"Non-numeric values: {df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]['TotalCharges'].unique()}")
