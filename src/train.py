# src/train.py
import pandas as pd
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor # <--- Changed to LightGBM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib # For saving the model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Import preprocessing function from the same package
from src.preprocess import load_data, preprocess_data

def train_model(X_train, y_train, preprocessor):
    """
    Trains a LightGBM Regressor model.
    """
    # Create a pipeline with preprocessing and the regressor
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', LGBMRegressor(random_state=42, n_jobs=-1))])
    # You can add parameters like n_estimators, learning_rate here if needed for tuning
    # e.g., LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)

    print("Training LightGBM Regressor model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")
    return model_pipeline

def evaluate_model(model, X_test, y_test, plot_results=True):
    """
    Evaluates the trained model and prints performance metrics.
    Optionally plots actual vs. predicted salaries.
    """
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2): {r2:.2f}')

    if plot_results:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Ideal line
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.title('Actual vs. Predicted Salary (LightGBM)')
        plt.grid(True)
        plt.savefig('actual_vs_predicted_salary_lightgbm.png')
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=50)
        plt.title('Distribution of Residuals (LightGBM)')
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('residuals_distribution_lightgbm.png')
        plt.show()

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

def save_model(model, path='models/lightgbm_model.pkl'): # <--- Updated model filename
    """Saves the trained model to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    data_path = '../data/Dataset_final.csv'
    df = load_data(data_path)

    if df is not None:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

        if X_train is not None:
            model = train_model(X_train, y_train, preprocessor)
            metrics = evaluate_model(model, X_test, y_test)
            save_model(model)
    else:
        print("Could not proceed with training due to data loading error.")