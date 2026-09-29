"""
Unit tests for model training and prediction.
"""

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import clean_data, prepare_features
from src.model import train_model, predict, predict_proba

def test_train_model():
    """Test that model trains successfully."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    X, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    assert model is not None, "Model should be trained"

def test_predictions_valid():
    """Test that predictions are valid."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    X, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    
    # Predictions should only be 0 or 1
    assert set(y_pred).issubset({0, 1}), "Predictions should be 0 or 1"

def test_predict_proba_valid():
    """Test that prediction probabilities are valid."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    X, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    y_proba = predict_proba(model, X_test)
    
    # Probabilities should be between 0 and 1
    assert (y_proba >= 0).all() and (y_proba <= 1).all(), "Probabilities should be between 0 and 1"
