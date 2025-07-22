# src/predict.py
import joblib
import pandas as pd
import os

def load_model(path='models/lightgbm_model.pkl'): # <--- Ensure this path is correct
    """Loads a trained model from a file."""
    try:
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}. Please ensure the model has been trained and saved.")
        return None

def predict_salary(model, new_data):
    """
    Makes salary predictions using the loaded model.
    `new_data` should be a pandas DataFrame with the same columns as the training data (excluding Salary).
    """
    if model is None:
        print("Model not loaded. Cannot make predictions.")
        return None

    print("Making predictions...")
    predictions = model.predict(new_data)
    return predictions

if __name__ == '__main__':
    # Example usage:
    new_employee_data = pd.DataFrame([{
        'Employment_Type': 'full-time',
        'Job_Role': 'data scientist',
        'Location': 'bangalore',
        'Company_Location': 'new york',
        'Remote_Ratio': 0,
        'Company_Size_Category': 'S',
        'Experience_Years': 5,
        'Age_Years': 28,
        'Gender': 'Male',
        'Education_Level': "Master's Degree",
        'Performance_Rating': 4.0,
        'Salaries_Reported_Count': 10 # Added this column back
    }])

    model = load_model()
    if model:
        predicted_salary = predict_salary(model, new_employee_data)
        if predicted_salary is not None:
            print(f"Predicted Salary for new employee: ${predicted_salary[0]:.2f}")