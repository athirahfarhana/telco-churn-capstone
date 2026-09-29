"""
Evaluate model performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Load model
with open("output/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nMetrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted No    Predicted Yes")
print(f"Actual No       {cm[0][0]:>6d}           {cm[0][1]:>6d}")
print(f"Actual Yes      {cm[1][0]:>6d}           {cm[1][1]:>6d}")

# Interpretation
tn, fp, fn, tp = cm.ravel()
print(f"\nInterpretation:")
print(f"True Negatives (correctly predicted no churn):  {tn}")
print(f"False Positives (predicted churn, didn't):      {fp}")
print(f"False Negatives (predicted no churn, did):      {fn}")
print(f"True Positives (correctly predicted churn):     {tp}")

print(f"\nBusiness Interpretation:")
print(f"- Of customers we predicted would churn: {tp / (tp + fp) * 100:.1f}% actually churned (Precision)")
print(f"- Of customers who actually churned: {tp / (tp + fn) * 100:.1f}% we caught (Recall)")

# Save metrics
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'TP': int(tp),
    'TN': int(tn),
    'FP': int(fp),
    'FN': int(fn)
}

import json
with open("output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to output/metrics.json")
