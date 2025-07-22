# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(file_path):
    """Loads the dataset from a given CSV file path."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

def preprocess_data(df, target_column='Salary'):
    """
    Preprocesses the dataframe:
    - Drops 'Employee_ID' and 'Company_Name' columns.
    - **Performs data cleaning/standardization on categorical columns.**
    - Separates features (X) and target (y).
    - Identifies numerical and categorical features.
    - Creates a preprocessing pipeline for scaling numerical and one-hot encoding categorical features.
    - Splits data into training and testing sets.
    """
    if df is None:
        return None, None, None, None, None

    # Drop columns not needed for modeling
    df = df.drop(columns=['Employee_ID', 'Company_Name'], errors='ignore')

    # --- START OF CORE DATA CLEANING (FOR TRAINING AND PREDICTION CONSISTENCY) ---
    # Apply cleaning to relevant categorical columns
    
    # Clean 'Employment_Type'
    if 'Employment_Type' in df.columns:
        df['Employment_Type'] = df['Employment_Type'].astype(str).str.strip().str.lower()
        df['Employment_Type'] = df['Employment_Type'].replace({
            'full time': 'full-time',
            'contract': 'contractor',
            'contracontractor': 'contractor' # Correcting typo
        })

    # Clean other object columns by stripping spaces and converting to lowercase
    # This ensures consistency for OneHotEncoder
    for col in ['Job_Role', 'Location', 'Company_Location', 'Gender', 'Education_Level']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Specific handling for 'Company_Size_Category' to keep 'S', 'M', 'L', 'E' uppercase
    if 'Company_Size_Category' in df.columns:
        df['Company_Size_Category'] = df['Company_Size_Category'].astype(str).str.strip().str.upper()

    # --- END OF CORE DATA CLEANING ---

    # Separate target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Create a preprocessor using ColumnTransformer
    # Numerical features will be scaled
    # Categorical features will be One-Hot Encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == '__main__':
    # Example usage (for testing this module independently)
    data_path = '../data/Dataset_final.csv'
    df = load_data(data_path)
    if df is not None:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        if X_train is not None:
            print("Data preprocessing complete.")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"y_test shape: {y_test.shape}")