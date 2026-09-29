"""
Model evaluation module.
"""

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    return metrics

def get_confusion_matrix(y_true, y_pred):
    """Get confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}

def save_metrics(metrics, cm_dict, filepath):
    """Save metrics to JSON file."""
    output = {'metrics': metrics, 'confusion_matrix': cm_dict}
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

def print_metrics(metrics, cm_dict):
    """Print metrics in readable format."""
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {cm_dict['TP']}, TN: {cm_dict['TN']}")
    print(f"FP: {cm_dict['FP']}, FN: {cm_dict['FN']}")
