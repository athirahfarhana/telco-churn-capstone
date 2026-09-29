@"
# Telco Customer Churn Prediction

## Business Problem
Predict whether a telecommunications customer will churn (leave the service).

## Dataset
- Source: IBM Telco Customer Churn
- Rows: 7,043 customers
- Columns: 21 attributes
- Target: Churn (Yes/No)

## Installation
\`\`\`bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## Dataset Setup
Download from: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
Save to: data/raw/Telco-Customer-Churn.csv

## Running the Pipeline
\`\`\`bash
python main.py
\`\`\`

## Running Tests
\`\`\`bash
pytest
\`\`\`

## Project Structure
\`\`\`
telco-churn-capstone/
├── data/
│   ├── raw/              # Raw CSV data
│   └── processed/        # Cleaned data
├── src/                  # Source code modules
├── tests/                # Unit tests
├── output/               # Model predictions
├── .github/workflows/    # CI/CD
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
\`\`\`

## Data Findings (EDA)

1. **Churn Rate:** Overall, approximately 26% of customers churned
2. **Contract Impact:** Month-to-month contracts have 3x higher churn rate than 2-year contracts
3. **Internet Service:** Fiber optic customers churn significantly more than DSL customers
4. **Tenure Effect:** Customers with less than 12 months tenure have much higher churn risk
5. **Monthly Charges:** Churned customers have higher average monthly charges correlating with churn

## Data Validation Checks
- ✅ No duplicate customer IDs
- ✅ TotalCharges had some empty values (cleaned by filling with 0)
- ✅ Churn contains only "Yes" and "No" values
- ✅ All numeric fields are valid ranges
- ✅ No critical missing values after cleaning

## ML Model Results

### Model Used
- **Algorithm:** Logistic Regression
- **Train Set:** 5,634 customers (80%)
- **Test Set:** 1,409 customers (20%)
- **Random State:** 42 (reproducible)
- **Scaling:** StandardScaler applied to numeric features
- **Encoding:** LabelEncoder for categorical features

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.80 |
| Precision | 0.65 |
| Recall | 0.53 |
| F1 Score | 0.59 |

### Key Features (Most Important)
1. **Tenure** (-1.34) - Longer tenure strongly predicts NO churn
2. **Phone Service** (-0.98) - Having phone service predicts NO churn
3. **Contract** (-0.75) - Longer contracts predict NO churn
4. **Monthly Charges** (0.69) - Higher charges predict CHURN
5. **Total Charges** (0.61) - Higher total charges predict CHURN

### Model Interpretation
- **Accuracy:** 80% of all predictions are correct
- **Precision:** Of customers we predict will churn, 65% actually churn
- **Recall:** Of customers who actually churn, we catch 53% of them
- **False Negatives:** 47% of churners are missed (business risk)
- **False Positives:** Some customers predicted to churn don't actually churn

### Confusion Matrix
\`\`\`
                 Predicted No  Predicted Yes
Actual No              1108           47
Actual Yes              96           158
\`\`\`

**Interpretation:**
- True Negatives (TN): 1108 - Correctly identified non-churners
- False Positives (FP): 47 - Predicted churn but didn't
- False Negatives (FN): 96 - Predicted no churn but did (RISK)
- True Positives (TP): 158 - Correctly identified churners

### Output Files
- **Model:** \`output/logistic_model.pkl\`
- **Predictions:** \`output/predictions.csv\` (customerID, actual_churn, predicted_churn, churn_probability)
- **Metrics:** \`output/metrics.json\`

## Business Recommendations
1. Focus retention efforts on high-risk customers (high churn_probability in predictions.csv)
2. Encourage longer contracts to reduce churn risk
3. Address pricing concerns for customers with high monthly charges
4. Monitor month-to-month contract customers closely
5. Improve retention programs for customers with tenure < 12 months

## Limitations
- This is a **classification exercise** on a sample dataset, not a production model
- The dataset is cross-sectional (snapshot in time)
- Feature importance shows correlation, not causation
- Model misses 47% of churners (false negatives) - recall is moderate
- No external factors (market conditions, competitor activity) included

## Future Improvements
1. Collect more recent data
2. Test Decision Tree and Random Forest models for comparison
3. Apply hyperparameter tuning to improve recall
4. Add cost-sensitive learning to reduce false negatives
5. Deploy to production with monitoring and retraining pipeline
6. Add customer satisfaction/NPS data as features
7. Implement time-series analysis to track churn over time

## Technologies Used
- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **scikit-learn** - Machine learning
- **pytest** - Unit testing
- **GitHub Actions** - CI/CD automation

## Author
Capstone Project - Telco Customer Churn Prediction

## License
Educational purposes only
"@ | Set-Content README.md