"""
End-to-end pipeline orchestration.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import clean_data, prepare_features
from src.model import train_model, save_model, predict, predict_proba
from src.evaluation import evaluate_model, get_confusion_matrix, save_metrics

def run_pipeline(csv_path, output_dir="output"):
    """
    Run the complete pipeline:
    Load -> Clean -> Prepare -> Train -> Evaluate -> Predict
    """
    print("=" * 60)
    print("TELCO CUSTOMER CHURN ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data(csv_path)
    print(f"    Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Clean data
    print("[2] Cleaning data...")
    df_clean = clean_data(df)
    print(f"    Data cleaned")
    
    # Prepare features
    print("[3] Preparing features...")
    X, y = prepare_features(df_clean)
    print(f"    Features shape: {X.shape}")
    
    # Split data
    print("[4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train model
    print("[5] Training model...")
    model = train_model(X_train, y_train)
    save_model(model, f"{output_dir}/logistic_model.pkl")
    print(f"    Model trained and saved")
    
    # Evaluate model
    print("[6] Evaluating model...")
    y_pred = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    cm_dict = get_confusion_matrix(y_test, y_pred)
    save_metrics(metrics, cm_dict, f"{output_dir}/metrics.json")
    print(f"    Accuracy: {metrics['Accuracy']:.4f}")
    print(f"    Precision: {metrics['Precision']:.4f}")
    print(f"    Recall: {metrics['Recall']:.4f}")
    
    # Generate predictions
    print("[7] Generating predictions...")
    y_pred_proba = predict_proba(model, X_test)
    df_original = load_data(csv_path)
    test_indices = y_test.index
    predictions_df = pd.DataFrame({
        'customerID': df_original.iloc[test_indices]['customerID'].values,
        'actual_churn': y_test.values,
        'predicted_churn': y_pred,
        'churn_probability': y_pred_proba
    })
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    print(f"    Predictions saved")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
