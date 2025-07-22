# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MODEL_PATH = 'models/lightgbm_model.pkl' # This path should point to your saved LightGBM model
DATA_PATH = 'data/Dataset_final.csv' # To get unique values for dropdowns and for distribution plots

# --- Load Model and Data for Dropdowns ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_trained_model(model_path):
    """Loads the trained model."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}. Please run `python run_project.py` first to train and save the model.")
        st.stop() # Stop the app if model is not found
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data # Cache data loading for dropdown options
def load_data_for_options(data_path):
    """
    Loads the original dataset to extract unique categorical values and for distribution plots.
    Applies display-specific formatting after core cleaning is assumed to be done in preprocess.py.
    """
    if not os.path.exists(data_path):
        st.error(f"Error: Dataset file not found at {data_path}.")
        st.stop()
    try:
        df = pd.read_csv(data_path)
        # Drop columns that were dropped during training for consistency
        df = df.drop(columns=['Employee_ID', 'Company_Name'], errors='ignore')

        # --- START OF UI-SPECIFIC DATA FORMATTING AND FILTERING ---
        # This section is for making dropdowns look clean and consistent in the UI.
        # The actual data used for model training should be cleaned in src/preprocess.py
        # We will apply .title() and specific replacements for display purposes.

        # Apply Title Case and strip spaces for display in UI for common string columns
        for col in ['Employment_Type', 'Job_Role', 'Location', 'Company_Location', 'Gender']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                # Specific handling for 'Full-time' to ensure hyphen for display
                if col == 'Employment_Type':
                    df[col] = df[col].replace('Full Time', 'Full-Time')
                    df[col] = df[col].replace('Contractor', 'Contractor') # Ensure consistent display
                    df[col] = df[col].replace('Contract', 'Contractor') # Ensure consistent display
                    df[col] = df[col].replace('Contracontractor', 'Contractor') # Ensure consistent display
        
        # Specific handling for 'Education_Level' to preserve 'Ph.D' and title case others
        if 'Education_Level' in df.columns:
            df['Education_Level'] = df['Education_Level'].astype(str).str.strip()
            df['Education_Level'] = df['Education_Level'].replace({
                'bachelor\'s degree': 'Bachelor\'s Degree',
                'master\'s degree': 'Master\'s Degree',
                'ph.d': 'Ph.D',
                'diploma': 'Diploma',
                'high school': 'High School',
                'some college': 'Some College' # Example for other levels if present
            })
            # For any other unique education levels not explicitly mapped, try title case
            df['Education_Level'] = df['Education_Level'].apply(
                lambda x: x if x in ['Bachelor\'s Degree', 'Master\'s Degree', 'Ph.D', 'Diploma', 'High School', 'Some College'] else x.title()
            )

        # Specific handling for 'Company_Size_Category' to keep 'S', 'M', 'L', 'E' uppercase
        if 'Company_Size_Category' in df.columns:
            df['Company_Size_Category'] = df['Company_Size_Category'].astype(str).str.strip().str.upper()

        # --- NEW: Filter out two-letter location codes for display ---
        for col in ['Location', 'Company_Location']:
            if col in df.columns:
                # Convert to string, strip, then filter out entries that are exactly 2 characters long
                df[col] = df[col].astype(str).str.strip()
                # Keep if length > 2 AND not empty string (to handle potential empty strings after strip)
                df = df[df[col].apply(lambda x: len(x) > 2 or x == '')] 
        # --- END OF UI-SPECIFIC DATA FORMATTING AND FILTERING ---

        return df
    except Exception as e:
        st.error(f"Error loading data for options: {e}")
        st.stop()

# Load model and data
model = load_trained_model(MODEL_PATH)
original_df = load_data_for_options(DATA_PATH)

# Get unique values for dropdowns from the original data
if original_df is not None:
    employment_types = original_df['Employment_Type'].unique().tolist()
    job_roles = original_df['Job_Role'].unique().tolist()
    locations = original_df['Location'].unique().tolist()
    company_locations = original_df['Company_Location'].unique().tolist()
    company_sizes = original_df['Company_Size_Category'].unique().tolist()
    genders = original_df['Gender'].unique().tolist()
    education_levels = original_df['Education_Level'].unique().tolist()
    remote_ratios = original_df['Remote_Ratio'].unique().tolist()
    # Sort for better user experience
    employment_types.sort()
    job_roles.sort()
    locations.sort()
    company_locations.sort()
    company_sizes.sort()
    genders.sort()
    education_levels.sort()
    remote_ratios.sort()
else:
    st.warning("Could not load original data for dropdown options. Using default options.")
    employment_types = ['Full-Time', 'Contractor', 'Internship', 'Part-Time', 'Freelance', 'Trainee'] # Updated defaults for UI
    job_roles = ['Data Scientist', 'Software Engineer', 'Data Analyst'] # Placeholder
    locations = ['Bangalore', 'Mumbai', 'New York', 'London'] # Placeholder
    company_locations = ['Bangalore', 'Mumbai', 'New York', 'London'] # Placeholder
    company_sizes = ['S', 'M', 'L', 'E']
    genders = ['Male', 'Female', 'Other']
    education_levels = ["Bachelor's Degree", "Master's Degree", "Ph.D", "Diploma", "High School"]
    remote_ratios = [0, 50, 100]


# --- Helper function to create input DataFrame for prediction ---
def create_prediction_input_df(
    employment_type, job_role, experience_years, age_years, gender,
    location, company_location, remote_ratio, company_size_category,
    education_level, performance_rating, salaries_reported_count
):
    """
    Creates a pandas DataFrame from input values for prediction.
    It passes the UI values directly, as the preprocessing pipeline (model)
    is now expected to handle the core cleaning and standardization.
    """
    return pd.DataFrame([{
        'Employment_Type': employment_type,
        'Job_Role': job_role,
        'Location': location,
        'Company_Location': company_location,
        'Remote_Ratio': remote_ratio,
        'Company_Size_Category': company_size_category,
        'Experience_Years': experience_years,
        'Age_Years': age_years,
        'Gender': gender,
        'Education_Level': education_level,
        'Performance_Rating': performance_rating,
        'Salaries_Reported_Count': salaries_reported_count
    }])


# --- Streamlit App Layout ---
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

st.title("💰 Employee Salary Prediction")
st.markdown("Enter employee details to predict their estimated annual salary.")

# Initialize session state for storing current form values and prediction status
if 'current_form_values' not in st.session_state:
    st.session_state.current_form_values = {
        'employment_type': employment_types[0] if employment_types else '',
        'job_role': job_roles[0] if job_roles else '',
        'experience_years': 5,
        'age_years': 30,
        'gender': genders[0] if genders else '',
        'location': locations[0] if locations else '',
        'company_location': company_locations[0] if company_locations else '',
        'remote_ratio': remote_ratios[0] if remote_ratios else 0,
        'company_size_category': company_sizes[0] if company_sizes else '',
        'education_level': education_levels[0] if education_levels else '',
        'performance_rating': 3.0,
        'salaries_reported_count': 10
    }
# NEW: Flag to track if a main prediction has been made
if 'main_prediction_made' not in st.session_state:
    st.session_state.main_prediction_made = False


# Create input fields for user
with st.form("salary_prediction_form"):
    st.header("Employee Details")

    col1, col2 = st.columns(2)
    with col1:
        # Ensure the index is valid for selectbox
        emp_type_idx = employment_types.index(st.session_state.current_form_values['employment_type']) if st.session_state.current_form_values['employment_type'] in employment_types else 0
        employment_type = st.selectbox("Employment Type", employment_types, key="main_employment_type", index=emp_type_idx)
        
        job_role_idx = job_roles.index(st.session_state.current_form_values['job_role']) if st.session_state.current_form_values['job_role'] in job_roles else 0
        job_role = st.selectbox("Job Role", job_roles, key="main_job_role", index=job_role_idx)
        
        experience_years = st.number_input("Experience (Years)", min_value=0, max_value=60, value=st.session_state.current_form_values['experience_years'], key="main_experience_years")
        age_years = st.number_input("Age (Years)", min_value=18, max_value=90, value=st.session_state.current_form_values['age_years'], key="main_age_years")
        
        gender_idx = genders.index(st.session_state.current_form_values['gender']) if st.session_state.current_form_values['gender'] in genders else 0
        gender = st.selectbox("Gender", genders, key="main_gender", index=gender_idx)

    with col2:
        location_idx = locations.index(st.session_state.current_form_values['location']) if st.session_state.current_form_values['location'] in locations else 0
        location = st.selectbox("Location (Employee)", locations, key="main_location", index=location_idx)
        
        company_location_idx = company_locations.index(st.session_state.current_form_values['company_location']) if st.session_state.current_form_values['company_location'] in company_locations else 0
        company_location = st.selectbox("Company Location", company_locations, key="main_company_location", index=company_location_idx)
        
        remote_ratio_idx = remote_ratios.index(st.session_state.current_form_values['remote_ratio']) if st.session_state.current_form_values['remote_ratio'] in remote_ratios else 0
        remote_ratio = st.selectbox("Remote Ratio (%)", remote_ratios, key="main_remote_ratio", index=remote_ratio_idx)
        
        company_size_category_idx = company_sizes.index(st.session_state.current_form_values['company_size_category']) if st.session_state.current_form_values['company_size_category'] in company_sizes else 0
        company_size_category = st.selectbox("Company Size Category", company_sizes, key="main_company_size_category", index=company_size_category_idx)
        
        education_level_idx = education_levels.index(st.session_state.current_form_values['education_level']) if st.session_state.current_form_values['education_level'] in education_levels else 0
        education_level = st.selectbox("Education Level", education_levels, key="main_education_level", index=education_level_idx)
        
        performance_rating = st.number_input("Performance Rating (0-5)", min_value=0.0, max_value=5.0, value=st.session_state.current_form_values['performance_rating'], step=0.1, key="main_performance_rating")
        salaries_reported_count = st.number_input("Salaries Reported Count", min_value=1, value=st.session_state.current_form_values['salaries_reported_count'], key="main_salaries_reported_count")

    # Prediction button
    submitted = st.form_submit_button("Predict Salary")

    if submitted:
        # Update session state with submitted values
        st.session_state.current_form_values = {
            'employment_type': employment_type,
            'job_role': job_role,
            'experience_years': experience_years,
            'age_years': age_years,
            'gender': gender,
            'location': location,
            'company_location': company_location,
            'remote_ratio': remote_ratio,
            'company_size_category': company_size_category,
            'education_level': education_level,
            'performance_rating': performance_rating,
            'salaries_reported_count': salaries_reported_count
        }
        st.session_state.main_prediction_made = True # Set flag to True after a prediction

        input_data = create_prediction_input_df(**st.session_state.current_form_values)

        try:
            predicted_salary = model.predict(input_data)[0]
            st.success(f"### Predicted Annual Salary: ${predicted_salary:,.2f}")
            st.info("Note: This is an estimated salary based on the trained model and provided inputs.")
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

st.markdown("---")

# --- Interactive What-If Scenario Analysis ---
# Only show What-If section if a main prediction has been made
if st.session_state.main_prediction_made:
    st.header("🔮 What-If Scenario Analysis")
    st.markdown("Adjust key factors to see how the predicted salary changes, based on your last input.")

    # Get current values from the main form's session state
    base_exp = st.session_state.current_form_values['experience_years']
    base_perf = st.session_state.current_form_values['performance_rating']

    col_what_if_1, col_what_if_2 = st.columns(2)

    with col_what_if_1:
        # Slider for Experience Years
        what_if_experience = st.slider(
            "Adjust Experience (Years)",
            min_value=0,
            max_value=60,
            value=base_exp,
            step=1,
            key="what_if_exp"
        )

    with col_what_if_2:
        # Slider for Performance Rating
        what_if_performance = st.slider(
            "Adjust Performance Rating (0-5)",
            min_value=0.0,
            max_value=5.0,
            value=base_perf,
            step=0.1,
            key="what_if_perf"
        )

    # Create input data for what-if scenario, using current form values as base
    # and overriding experience/performance from sliders
    what_if_input_values = st.session_state.current_form_values.copy()
    what_if_input_values['experience_years'] = what_if_experience
    what_if_input_values['performance_rating'] = what_if_performance

    what_if_input_df = create_prediction_input_df(**what_if_input_values)

    # Make real-time prediction for what-if scenario
    try:
        what_if_predicted_salary = model.predict(what_if_input_df)[0]
        st.metric(label="What-If Predicted Salary", value=f"${what_if_predicted_salary:,.2f}")
    except Exception as e:
        st.warning(f"Could not predict for what-if scenario: {e}")
else:
    st.info("Submit the main 'Predict Salary' form above to enable the 'What-If Scenario Analysis'.")


st.markdown("---")

# --- Original Feature: Salary Distribution by Category ---
st.header("📊 Salary Distribution Insights")
st.markdown("Explore the salary distribution for different categories in the dataset.")

if original_df is not None:
    # Identify categorical columns available in the original_df for plotting
    # Exclude 'Salary' itself and any IDs/names
    plot_categorical_cols = [col for col in original_df.select_dtypes(include='object').columns if col not in ['Employee_ID', 'Company_Name']]
    
    selected_category_for_plot = st.selectbox(
        "Select a category to view salary distribution:",
        plot_categorical_cols
    )

    if selected_category_for_plot:
        st.subheader(f"Salary Distribution by {selected_category_for_plot}")
        
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        unique_values_count = original_df[selected_category_for_plot].nunique()

        if unique_values_count > 20: # Arbitrary threshold for too many categories for a direct boxplot
            # For many categories, show top N categories by average salary
            top_n = 10
            avg_salary_by_category = original_df.groupby(selected_category_for_plot)['Salary'].mean().nlargest(top_n).index
            df_filtered = original_df[original_df[selected_category_for_plot].isin(avg_salary_by_category)]
            
            sns.boxplot(x='Salary', y=selected_category_for_plot, data=df_filtered, ax=ax, palette='viridis',
                        order=avg_salary_by_category) # Order by average salary
            ax.set_title(f'Top {top_n} Categories by Average Salary in {selected_category_for_plot}')
            ax.set_xlabel('Salary')
            ax.set_ylabel(selected_category_for_plot)
        else:
            # For fewer categories, a standard violin plot is good for showing distribution shape
            sns.violinplot(x='Salary', y=selected_category_for_plot, data=original_df, ax=ax, palette='viridis')
            ax.set_title(f'Salary Distribution Across {selected_category_for_plot} Categories')
            ax.set_xlabel('Salary')
            ax.set_ylabel(selected_category_for_plot)
        
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory

else:
    st.info("Data for distribution insights is not available.")

st.markdown("---")
st.markdown("Developed for IBM & Edunet Foundation AI/ML Internship Capstone Project")
