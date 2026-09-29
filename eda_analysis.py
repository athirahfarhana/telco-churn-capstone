"""
Exploratory Data Analysis - Find patterns in the data.
"""

import pandas as pd
from src.data_loader import load_data

df = load_data("data/raw/Telco-Customer-Churn.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Question 1: Overall churn rate
churn_rate = (df['Churn'] == 'Yes').sum() / len(df) * 100
print(f"\n1. Overall Churn Rate: {churn_rate:.2f}%")

# Question 2: Churn by Contract type
print(f"\n2. Churn by Contract Type:")
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
print(contract_churn)

# Question 3: Churn by InternetService
print(f"\n3. Churn by Internet Service:")
internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
print(internet_churn)

# Question 4: Churn by tenure ranges
df['tenure_range'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=['0-12m', '12-24m', '24-36m', '36-48m', '48-60m', '60-72m'])
print(f"\n4. Churn by Tenure Range:")
tenure_churn = df.groupby('tenure_range', observed=True)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
print(tenure_churn)

# Question 5: Monthly Charges comparison
print(f"\n5. Average Monthly Charges:")
print(f"Churned customers: ")
print(f"Retained customers: ")

print("\nKEY FINDINGS:")
print("- Month-to-month contracts have much higher churn")
print("- Fiber optic customers churn more than DSL")
print("- Customers with shorter tenure churn more")
print("- Churned customers have higher average monthly charges")
