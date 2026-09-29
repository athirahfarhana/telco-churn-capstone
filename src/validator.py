"""
Data validation module.
"""

import pandas as pd

def validate_dataset(df):
    """
    Validate the raw dataset.
    Returns a report of validation checks.
    """
    report = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'duplicate_ids': df['customerID'].duplicated().sum(),
        'missing_values': df.isna().sum().to_dict(),
        'churn_values': df['Churn'].unique().tolist()
    }
    return report

def validate_churn_values(df):
    """Validate that Churn column contains only Yes/No."""
    valid_values = {'Yes', 'No'}
    actual_values = set(df['Churn'].unique())
    return actual_values == valid_values

def validate_no_duplicates(df):
    """Validate that there are no duplicate customer IDs."""
    return df['customerID'].duplicated().sum() == 0

def validate_numeric_columns(df, columns):
    """Validate that specified columns are numeric."""
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False
    return True
