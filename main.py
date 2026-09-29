"""
Telco Customer Churn Prediction - Main Entry Point
"""

from src.pipeline import run_pipeline

def main():
    # Run the complete pipeline
    run_pipeline("data/raw/Telco-Customer-Churn.csv", output_dir="output")

if __name__ == "__main__":
    main()
