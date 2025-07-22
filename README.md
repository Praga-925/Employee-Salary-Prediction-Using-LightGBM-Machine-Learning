# Employee Salary Prediction using LightGBM Machine Learning and Interactive Data Insights

This project aims to develop a robust machine learning model to predict employee salaries based on various features, and to provide interactive data insights through a user-friendly web application. This project was developed as a capstone for the IBM & Edunet Foundation AI/ML Internship.

## Project Structure

The project is organized into a modular and scalable structure:

employee_salary_prediction/
├── data/
│   └── Dataset_final.csv           # The raw dataset used for training and insights.
├── src/
│   ├── init.py                 # Makes 'src' a Python package.
│   ├── preprocess.py               # Contains functions for data loading, cleaning, and preprocessing.
│   ├── train.py                    # Handles the machine learning model training and evaluation.
│   └── predict.py                  # (Optional) Contains functions for making new predictions using the saved model.
├── models/
│   └── lightgbm_model.pkl          # The saved trained LightGBM model pipeline.
├── notebooks/
│   └── salary_prediction_eda.ipynb # Optional: Jupyter Notebook for exploratory data analysis and experimentation.
├── requirements.txt                # Lists all Python dependencies required for the project.
├── README.md                       # This project documentation file.
├── run_project.py                  # Main script to orchestrate the data preprocessing, model training, and saving.
└── app.py                          # The Streamlit web application for interactive prediction and insights.


## Problem Statement

In today's dynamic job market, accurately determining employee salaries is crucial for both businesses and individuals. Companies need fair compensation strategies for effective budgeting, talent acquisition, and employee retention. Simultaneously, employees benefit from understanding the various factors that influence their earning potential and career progression. Traditional methods of salary determination can often be subjective, leading to inconsistencies and a lack of data-driven insights. This project addresses the challenge of predicting employee salaries by leveraging advanced machine learning techniques, aiming to provide data-driven estimations and interactive insights into salary trends.

**Note:** This project utilizes a **simulated/synthetic dataset** for demonstration purposes. Therefore, the predicted salary values and observed salary distributions are based on the patterns within this simulated data and **may not reflect actual real-world salary trends or magnitudes.**

## System Development Approach (Technology Used)

**Overall Strategy:**
An end-to-end Machine Learning (ML) pipeline approach was adopted to ensure a structured, reproducible, and scalable workflow. This pipeline encompasses all stages from data acquisition and rigorous preprocessing to advanced model training, comprehensive evaluation, and interactive web application deployment.

**System Requirements:**
* **Operating System:** Windows, macOS, or Linux
* **Software:** Python 3.8+
* **Environment Management:** Virtual environment (e.g., `venv` or `conda`) is highly recommended to manage project dependencies.
* **Development Environment:** Visual Studio Code, Jupyter Notebooks, or any compatible Integrated Development Environment (IDE).
* **User Interface:** A modern web browser is required to interact with the Streamlit application.

**Libraries Required to Build the Model and Application:**
* **`pandas`**: Essential for efficient data loading, manipulation, and cleaning operations.
* **`scikit-learn`**: Utilized for fundamental machine learning tasks including data splitting (train/test sets), robust preprocessing techniques (StandardScaler for numerical features, OneHotEncoder for categorical features), and constructing the overall ML `Pipeline`.
* **`lightgbm`**: The core machine learning algorithm, specifically `LGBMRegressor`, chosen for its high performance, speed, and efficiency in handling large tabular datasets for regression tasks.
* **`matplotlib` & `seaborn`**: Powerful libraries for creating static and statistical data visualizations, used extensively for model evaluation plots and generating data insights within the Streamlit application.
* **`streamlit`**: The open-source Python framework used to rapidly build and deploy the interactive and user-friendly web application (front-end) for the project.
* **`joblib`**: Employed for saving the trained machine learning model pipeline to disk and loading it back for predictions, ensuring model persistence across sessions.
* **`numpy`**: Provides essential numerical computing capabilities, often used implicitly by other libraries like pandas and scikit-learn.

## Algorithm & Deployment (Step by Step Procedure)

The project follows a systematic procedure to ensure robust development and deployment:

1.  **Data Loading & Initial Inspection:**
    * The project initiates by loading the `Dataset_final.csv` file into a pandas DataFrame.
    * An initial inspection is performed to understand the dataset's structure, data types, identify any missing values, and view basic descriptive statistics of the features.

2.  **Data Preprocessing & Cleaning:**
    * **Irrelevant Column Removal:** Columns such as `Employee_ID` (a unique identifier) and `Company_Name` (highly granular, potentially noisy for general prediction) were dropped.
    * **Robust Categorical Data Cleaning:**
        * **Standardization:** Inconsistent entries within categorical features (e.g., "full time", "full-time", "Full Time" in `Employment_Type` were consolidated to a single "full-time" representation).
        * **Typo Correction:** Obvious typos and similar categories (e.g., "contracontractor" and "contract" were mapped to "contractor") were standardized.
        * **Formatting Consistency:** Leading/trailing spaces were removed, and consistent casing (lowercase for model training, Title Case for UI display) was applied across all relevant categorical features (`Job_Role`, `Location`, `Gender`, `Education_Level`, etc.).
        * **Location Filtering (UI-Specific):** Ambiguous 2-letter location codes (e.g., 'gb', 'us') were filtered out from the 'Location' and 'Company\_Location' dropdown options in the Streamlit UI for improved user clarity, while the model's preprocessing pipeline in `src/preprocess.py` handles all original data for training.
    * **Numerical Feature Scaling:** `StandardScaler` was applied to numerical features (`Experience_Years`, `Age_Years`, `Performance_Rating`, `Salaries_Reported_Count`) to transform their values to a common scale (mean 0, variance 1). This prevents features with larger numerical ranges from disproportionately influencing the model.
    * **Categorical Feature Encoding:** `OneHotEncoder` converted the cleaned categorical text features into a numerical binary format, which is a prerequisite for the LightGBM model.
    * **Data Splitting:** The preprocessed dataset was partitioned into an 80% training set (used for model learning) and a 20% unseen test set (reserved for unbiased performance evaluation).

3.  **Model Training (LightGBM Regressor):**
    * A `scikit-learn Pipeline` was constructed, seamlessly integrating the preprocessing steps (`ColumnTransformer`) with the `LGBMRegressor` model. This ensures that data flows through the correct transformations before reaching the model.
    * The `LGBMRegressor` was chosen as the core machine learning algorithm due to its superior speed, efficiency, and high accuracy on large tabular datasets. It operates by building an ensemble of decision trees sequentially, with each new tree designed to correct the prediction errors of the preceding ones.
    * The model was trained (`.fit()`) on the preprocessed training data (`X_train`, `y_train`).

4.  **Model Evaluation:**
    * The trained model's predictive performance was rigorously assessed on the held-out test set (`X_test`, `y_test`).
    * Key regression metrics were computed and analyzed:
        * **Mean Absolute Error (MAE):** The average absolute difference between predicted and actual salaries.
        * **Mean Squared Error (MSE):** The average of the squared differences, penalizing larger errors more heavily.
        * **Root Mean Squared Error (RMSE):** The square root of MSE, providing an error metric in the same units as the target variable (salary).
        * **R-squared (R2 Score):** Represents the proportion of the variance in the dependent variable (Salary) that is predictable from the independent variables (features).
    * Visualizations, including "Actual vs. Predicted Salary" scatter plots and "Distribution of Residuals" histograms, were generated to provide a graphical understanding of the model's performance and error patterns.

5.  **Model Persistence:**
    * The entire trained `Pipeline` object (which encapsulates both the fitted preprocessor and the `LGBMRegressor`) was saved to a `.pkl` file (`models/lightgbm_model.pkl`) using the `joblib` library. This allows the model to be loaded quickly for predictions without needing to retrain it every time the application runs.

6.  **Streamlit Web Application Deployment:**
    * An interactive web application (`app.py`) was developed using the Streamlit framework, serving as the user-friendly interface for the project.
    * The application loads the saved `lightgbm_model.pkl` for immediate use.
    * **Core Prediction Interface:** Provides intuitive input fields (dropdowns, number inputs) for users to enter employee details. Upon clicking "Predict Salary," the application uses the loaded model to generate and display an estimated annual salary.
    * **Unique Feature 1: Interactive What-If Scenario Analysis (Innovation Highlight):** This feature allows users to dynamically adjust key numerical factors like 'Experience Years' and 'Performance Rating' via interactive sliders. The predicted salary updates in real-time based on these adjustments, keeping all other factors constant. This provides intuitive insights into the direct impact of specific features on salary predictions and enables exploration of hypothetical career growth scenarios.
    * **Unique Feature 2: Salary Distribution Insights (Innovation Highlight):** This feature empowers users to select any categorical feature from the dataset (e.g., `Job_Role`, `Education_Level`, `Location`). It then visualizes the actual salary distribution for different categories within that selected feature (using informative box plots or violin plots). This provides valuable contextual understanding of salary ranges and variability within specific groups, complementing the single salary prediction.
    * The application's User Interface (UI) is designed for clarity, ease of interaction, and a professional appearance.

## Result
<img width="1919" height="991" alt="Screenshot 2025-07-22 183418" src="https://github.com/user-attachments/assets/a0401d0f-40c2-4622-b879-57b81e65944e" />
<img width="1919" height="1017" alt="Screenshot 2025-07-22 190027" src="https://github.com/user-attachments/assets/b0622158-2e52-4fe5-b8d7-0d4449e1a7ef" />
<img width="1919" height="1018" alt="Screenshot 2025-07-22 190016" src="https://github.com/user-attachments/assets/74ad4db7-a499-4475-b35a-f895329cf5b6" />
<img width="1919" height="1079" alt="Screenshot 2025-07-22 184156" src="https://github.com/user-attachments/assets/3819f622-b679-46e7-a56c-51a94067a12a" />
<img width="1919" height="1079" alt="Screenshot 2025-07-22 184126" src="https://github.com/user-attachments/assets/d0a24a3f-661b-4cbe-8e25-a092d6c0ecee" />



* **Console Output of Model Evaluation:**
    * *(Insert Screenshot of the terminal output showing MAE, MSE, RMSE, and R2 scores after `python run_project.py` completes.)*
    * *Brief Explanation:* "This output quantifies our LightGBM model's performance on unseen data. For example, an R2 score of 0.37 indicates that 37% of the variance in salary can be explained by our model's features."

* **Model Evaluation Plot 1: Actual vs. Predicted Salary:**
    * *(Insert Screenshot of the plot generated by `train.py` (e.g., `actual_vs_predicted_salary_lightgbm.png`).)*
    * *Brief Explanation:* "This scatter plot visually demonstrates the alignment between our model's predictions and actual salaries. Points clustered tightly around the red dashed line signify strong predictive accuracy."

* **Model Evaluation Plot 2: Distribution of Residuals:**
    * *(Insert Screenshot of the plot generated by `train.py` (e.g., `residuals_distribution_lightgbm.png`).)*
    * *Brief Explanation:* "The histogram of residuals, ideally centered around zero and normally distributed, indicates that our model's prediction errors are generally unbiased and random."

* **Streamlit App Screenshot 1: Main Prediction Interface:**
    * *(Insert Screenshot of the top part of your `app.py` running, showing the input form and a predicted salary displayed after submission.)*
    * *Brief Explanation:* "Our interactive web application provides a user-friendly interface for inputting employee details and receiving an instant salary prediction."

* **Streamlit App Screenshot 2: What-If Scenario Analysis:**
    * *(Insert Screenshot of the 'What-If Scenario Analysis' section with the sliders and the dynamically updating predicted salary.)*
    * *Brief Explanation:* "This unique feature empowers users to explore the direct impact of adjusting key factors like experience or performance on potential salary, offering actionable insights for career planning."

* **Streamlit App Screenshot 3: Salary Distribution Insights:**
    * *(Insert Screenshot of the 'Salary Distribution Insights' section with a selected category (e.g., 'Job Role') and its corresponding salary distribution plot.)*
    * *Brief Explanation:* "This feature provides valuable context by visualizing the actual salary ranges and distributions within different employee categories from the dataset, enhancing understanding of salary trends."

* **GitHub Repository Link:**
    * `https://github.com/your_username/your_repository_name` *(Please replace with your actual GitHub link)*

## Conclusion

* Successfully developed a robust, end-to-end Employee Salary Prediction system leveraging the highly efficient LightGBM algorithm.
* The model effectively captures underlying patterns within the provided dataset, demonstrating a moderate ability to predict salaries (R2 = 0.37).
* The Streamlit application provides an intuitive and interactive platform for salary prediction, significantly enhanced by the innovative "What-If Scenario Analysis" and "Salary Distribution Insights" features, which offer valuable contextual understanding and exploratory capabilities.
* The project showcases a practical application of machine learning for HR analytics, offering data-driven estimations and insights into compensation dynamics.

**Challenges Encountered During Implementation:**
* **Data Quality & Consistency:** A significant challenge involved cleaning and standardizing inconsistent and erroneous entries within categorical features (e.g., 'Employment\_Type', 'Location'), which was crucial for reliable model training and UI display.
* **Data Realism & Scale:** The synthetic nature of the dataset led to salary magnitudes that may not reflect real-world values, necessitating clear disclaimers within the application and documentation.
* **Initial Model Training Efficiency:** Early exploration with less optimized algorithms like Random Forest highlighted the importance of selecting highly efficient models like LightGBM for handling large datasets effectively within practical timeframes.

**Key Learnings:**
The project reinforced the critical importance of meticulous data preprocessing, the strategic selection of efficient machine learning algorithms for specific data characteristics, and the value of creating interactive, insightful user interfaces to make complex machine learning models accessible and understandable.

## Future Scope (Optional)

* **Hyperparameter Tuning:** Implement advanced hyperparameter optimization techniques (e.g., `GridSearchCV`, `RandomizedSearchCV`, Bayesian Optimization) for the LightGBM model to further fine-tune its parameters and potentially achieve even higher predictive accuracy.
* **Advanced Feature Engineering:** Explore creating more sophisticated features from existing data, such as interaction terms (e.g., `Experience_Years` * `Job_Role`), polynomial features, or deriving higher-level categorical features (e.g., 'Location Tier' based on average salaries in a city).
* **Real-World Data Integration:** Transition to or augment the current dataset with a larger, more diverse, and verified real-world salary dataset. This would significantly improve the model's generalizability and provide more accurate, actionable predictions for actual scenarios.
* **Model Interpretability (Advanced):** Integrate advanced interpretability techniques like SHAP (SHapley Additive exPlanations) values to provide granular, local explanations for *individual* salary predictions, enhancing transparency and trust in the model's outputs.
* **Deployment Scaling:** For a production-grade application, consider deploying the machine learning model as a separate microservice (e.g., using Flask or FastAPI) with a dedicated API, allowing for better scalability, maintainability, and integration with other systems, independent of the Streamlit UI.

## References

* **Python Libraries:**
    * Pandas: `https://pandas.pydata.org/`
    * Scikit-learn: `https://scikit-learn.org/`
    * LightGBM: `https://lightgbm.readthedocs.io/`
    * Matplotlib: `https://matplotlib.org/`
    * Seaborn: `https://seaborn.pydata.org/`
    * Streamlit: `https://streamlit.io/`
* **Dataset Source:** `Dataset_final.csv` (If this dataset was sourced from a public repository like Kaggle, please provide the specific URL here).
* *(Add any specific research papers, articles, or online courses that were instrumental in your learning or project development during the internship.)*

## THANK YOU
