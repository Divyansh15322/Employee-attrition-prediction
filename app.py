"""
Employee Attrition Prediction — Streamlit App
Loads the pre-trained Logistic Regression model + preprocessing artifacts
and serves real-time predictions through an interactive UI.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")

MODEL_DIR = "models"


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/model.pkl")
    encoder = joblib.load(f"{MODEL_DIR}/encoder.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.pkl")
    num_cols = joblib.load(f"{MODEL_DIR}/num_cols.pkl")
    cat_options = joblib.load(f"{MODEL_DIR}/cat_options.pkl")
    return model, encoder, scaler, feature_names, cat_cols, num_cols, cat_options


model, encoder, scaler, feature_names, cat_cols, num_cols, cat_options = load_artifacts()

st.title("📊 Employee Attrition Predictor")
st.caption(
    "Predicts the likelihood an employee will leave the company, using a "
    "Logistic Regression model trained on IBM HR Analytics data "
    "(87% accuracy, 0.70 precision on the attrition class)."
)

st.divider()
st.subheader("Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    business_travel = st.selectbox("Business Travel", cat_options["BusinessTravel"])
    department = st.selectbox("Department", cat_options["Department"])
    education_field = st.selectbox("Education Field", cat_options["EducationField"])
    gender = st.selectbox("Gender", cat_options["Gender"])
    job_role = st.selectbox("Job Role", cat_options["JobRole"])
    marital_status = st.selectbox("Marital Status", cat_options["MaritalStatus"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)

with col2:
    monthly_income = st.number_input("Monthly Income", 1000, 25000, 5000, step=100)
    job_level = st.slider("Job Level", 1, 5, 2)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
    total_working_years = st.slider("Total Working Years", 0, 40, 8)
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)

with st.expander("More details (optional — defaults are typical values)"):
    c1, c2 = st.columns(2)
    with c1:
        daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
        education = st.slider("Education Level (1-5)", 1, 5, 3)
        num_companies_worked = st.slider("Num Companies Worked", 0, 10, 2)
    with c2:
        percent_salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
        performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
        years_in_current_role = st.slider("Years In Current Role", 0, 20, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)

st.divider()

if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
    input_dict = {
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": 1 if overtime == "Yes" else 0,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    input_df = pd.DataFrame([input_dict])

    X_cat = encoder.transform(input_df[cat_cols]).toarray()
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
    X_num_df = input_df[num_cols]
    X_all = pd.concat([X_cat_df, X_num_df], axis=1)
    X_all = X_all[feature_names]  # ensure column order matches training

    X_scaled = scaler.transform(X_all)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk — {probability * 100:.1f}% probability of leaving")
    else:
        st.success(f"✅ Low Attrition Risk — {probability * 100:.1f}% probability of leaving")

    st.progress(min(int(probability * 100), 100))
    st.caption(
        "Model: Logistic Regression | Trained on 1,470 IBM HR records | "
        "Test accuracy: 87% | Precision (attrition class): 0.70"
    )

st.divider()
st.caption("Built with Streamlit · Scikit-learn · Docker")
