"""
Prepare features and split into train/test.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load cleaned data
df = pd.read_csv("data/processed/telco_cleaned.csv")

print("=" * 60)
print("FEATURE PREPARATION")
print("=" * 60)

# Check for NaN values
print(f"\nMissing values before cleaning:")
print(df.isna().sum())

# Drop rows with any NaN values
df = df.dropna()

print(f"\nAfter dropping NaN rows: {df.shape}")

# Remove customerID (not a feature)
df = df.drop(columns=['customerID'])

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

print(f"\nTarget (y) shape: {y.shape}")
print(f"Features (X) shape: {X.shape}")
print(f"Churn distribution:\n{y.value_counts()}")

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Encode categorical variables
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

print(f"\nCategorical variables encoded")

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"Numeric features scaled")

# Verify no NaN values remain
print(f"\nNaN values after encoding/scaling: {X.isna().sum().sum()}")

print(f"\nFeatures after encoding and scaling:")
print(X.head())

# Split into train and test (80/20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")

# Verify no NaN in train/test
print(f"\nNaN in X_train: {X_train.isna().sum().sum()}")
print(f"NaN in X_test: {X_test.isna().sum().sum()}")

# Save for next step
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nTrain/test split saved to data/processed/")
