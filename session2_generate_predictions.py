"""
Generate predictions CSV for output.
"""

import pandas as pd
import pickle
from src.data_loader import load_data

# Load original data (for customerID)
df_original = load_data("data/raw/Telco-Customer-Churn.csv")

# Load test data and actual churn values
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load model
with open("output/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

# Get test indices to match with original customerIDs
test_indices = y_test.index
customer_ids = df_original.iloc[test_indices]['customerID'].values
actual_churn = y_test.values.ravel()

# Make predictions
predicted_churn = model.predict(X_test)
churn_probability = model.predict_proba(X_test)[:, 1]

# Create output dataframe
predictions_df = pd.DataFrame({
    'customerID': customer_ids,
    'actual_churn': actual_churn,
    'predicted_churn': predicted_churn,
    'churn_probability': churn_probability
})

# Save to CSV
predictions_df.to_csv("output/predictions.csv", index=False)

print(f"\nPredictions generated:")
print(f"Total predictions: {len(predictions_df)}")
print(f"Predicted churners: {(predicted_churn == 1).sum()}")
print(f"Predicted non-churners: {(predicted_churn == 0).sum()}")

print(f"\nFirst 10 predictions:")
print(predictions_df.head(10))

print(f"\nPredictions saved to output/predictions.csv")
