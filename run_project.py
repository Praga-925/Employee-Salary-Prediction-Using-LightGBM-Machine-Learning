# run_project.py
import os
from src.preprocess import load_data, preprocess_data
from src.train import train_model, evaluate_model, save_model

def main():
    """Main function to run the salary prediction project."""
    print("Starting Employee Salary Prediction Project...")

    # Define data path
    data_file_path = 'data/Dataset_final.csv'

    # --- Step 1: Load Data ---
    df = load_data(data_file_path)
    if df is None:
        print("Project terminated due to data loading error.")
        return

    # --- Step 2: Preprocess Data ---
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    if X_train is None:
        print("Project terminated due to data preprocessing error.")
        return

    # --- Step 3: Train Model ---
    model = train_model(X_train, y_train, preprocessor)

    # --- Step 4: Evaluate Model ---
    evaluation_metrics = evaluate_model(model, X_test, y_test, plot_results=True)
    print("\nModel Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"- {metric.upper()}: {value:.2f}")

    # --- Step 5: Save Model ---
    save_model(model)

    print("\nProject execution complete!")
    print("Check 'models/' directory for the saved model and current directory for plots.")

if __name__ == '__main__':
    main()