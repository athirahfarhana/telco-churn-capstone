"""
Model training and prediction module.
"""

import pickle
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, random_state=42):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save model to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load model from pickle file."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, X):
    """Make predictions."""
    return model.predict(X)

def predict_proba(model, X):
    """Get prediction probabilities."""
    return model.predict_proba(X)[:, 1]
