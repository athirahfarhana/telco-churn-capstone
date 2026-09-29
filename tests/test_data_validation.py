"""
Unit tests for data validation.
"""

import pytest
import pandas as pd
from src.data_loader import load_data
from src.validator import validate_churn_values, validate_no_duplicates

def test_load_data():
    """Test that data loads correctly."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    assert df.shape[0] > 0, "Data should have rows"
    assert df.shape[1] > 0, "Data should have columns"

def test_churn_values_valid():
    """Test that Churn column contains only Yes/No."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    assert validate_churn_values(df), "Churn should only contain Yes/No"

def test_no_duplicate_ids():
    """Test that there are no duplicate customer IDs."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    assert validate_no_duplicates(df), "Customer IDs should be unique"

def test_required_columns_exist():
    """Test that required columns exist."""
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    required_cols = ['customerID', 'Churn', 'tenure', 'MonthlyCharges']
    for col in required_cols:
        assert col in df.columns, f"Column {col} should exist"
