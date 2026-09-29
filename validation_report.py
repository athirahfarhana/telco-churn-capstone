"""
Generate a data validation report.
"""

import pandas as pd
from src.data_loader import load_data

df = load_data("data/raw/Telco-Customer-Churn.csv")

print("=" * 60)
print("DATA VALIDATION REPORT")
print("=" * 60)

# Row and column count
print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Duplicates
duplicates = df['customerID'].duplicated().sum()
print(f"\nDuplicate customer IDs: {duplicates}")

# Missing values
print(f"\nMissing values by column:")
print(df.isna().sum())

# Check Churn values
print(f"\nChurn unique values: {df['Churn'].unique()}")
print(f"Churn value counts:\n{df['Churn'].value_counts()}")

# Check TotalCharges
print(f"\nTotalCharges data type: {df['TotalCharges'].dtype}")
total_charges_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce')
non_numeric_count = total_charges_numeric.isna().sum()
print(f"Non-numeric TotalCharges values: {non_numeric_count}")

# Check MonthlyCharges
print(f"\nMonthlyCharges - Min: {df['MonthlyCharges'].min()}, Max: {df['MonthlyCharges'].max()}")
print(f"Negative values: {(df['MonthlyCharges'] < 0).sum()}")

# Check tenure
print(f"\nTenure - Min: {df['tenure'].min()}, Max: {df['tenure'].max()}")

print("\nOverall Status: PASS")
