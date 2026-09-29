"""
Train Logistic Regression model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load train data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

print("=" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 60)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(f"\nModel trained successfully!")
print(f"Model parameters: {model.get_params()}")
print(f"Feature importance (coefficients):")
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(coefficients.head(10))

# Save model
with open("output/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to output/logistic_model.pkl")
