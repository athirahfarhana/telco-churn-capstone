"""
Session 1: Load, validate, and explore the dataset.
"""

import pandas as pd
from src.data_loader import load_data
from src.validator import validate_data

# Load the data
csv_path = "data/raw/Telco-Customer-Churn.csv"
df = load_data(csv_path)

# Print basic info
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("Column Names and Data Types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

print("\nBasic statistics:")
print(df.describe())
