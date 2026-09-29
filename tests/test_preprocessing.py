"""
Unit tests for preprocessing.
"""

import pytest
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_data, prepare_features, validate_features

def test_clean_data():
    """Test that data cleaning works."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    
    # Churn should be binary (0 or 1)
    assert set(df_clean['Churn'].unique()).issubset({0, 1})
    
    # TotalCharges should be numeric
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])

def test_prepare_features():
    """Test that features are prepared correctly."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    X, y = prepare_features(df_clean)
    
    # customerID should not be in features
    assert 'customerID' not in X.columns
    
    # Churn should not be in features
    assert 'Churn' not in X.columns
    
    # y should be non-null
    assert y is not None and len(y) > 0

def test_validate_features():
    """Test that features validation works."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    X, y = prepare_features(df_clean)
    
    # Should not raise exception
    assert validate_features(X, y) == True
