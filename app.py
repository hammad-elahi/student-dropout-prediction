import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Student Dropout Risk Predictor")

col1, col2 = st.columns(2)

with col1:
    approved_1st_sem = st.number_input("1st Semester Approved Units", min_value=0, max_value=26, value=5, help="Number of courses successfully passed in the 1st semester")
    grade_1st_sem = st.number_input("1st Semester Grade", min_value=0.0, max_value=20.0, value=12.0, step=0.5, help="Average grade in 1st semester (scale 0-20)")
    

with col2:
    approved_2nd_sem = st.number_input("2nd Semester Approved Units", min_value=0, max_value=26, value=5, help="Number of courses successfully passed in the 2nd semester")
    grade_2nd_sem = st.number_input("2nd Semester Grade", min_value=0.0, max_value=20.0, value=12.0, step=0.5, help="Average grade in 2nd semester (scale 0-20)")

age = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20 , help= "Student's age when they enrolled")

tuition_paid = st.checkbox("Tuition Fees Up to Date" , help= "Check if the student's tuition payment are current")

debtor = st.checkbox("Is Debtor" , help= "Check if the student currently owes money to the institution")

scholarship = st.checkbox("Scholarship Holder" , help= "Check if the student holds a scholarship")

if st.button("Predict Risk"):
    input_data = {
        "Marital status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 2,
        "Daytime/evening attendance": 1,
        "Previous qualification": 1,
        "Nacionality": 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 5,
        "Father's occupation": 5,
        "Displaced": 0,
        "Educational special needs": 0,
        "Debtor": int(debtor),
        "Tuition fees up to date": int(tuition_paid),
        "Gender": 1,
        "Scholarship holder": int(scholarship),
        "Age at enrollment": age,
        "International": 0,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": approved_1st_sem,
        "Curricular units 1st sem (grade)": grade_1st_sem,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": approved_2nd_sem,
        "Curricular units 2nd sem (grade)": grade_2nd_sem,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10.8,
        "Inflation rate": 1.4,
        "GDP": 1.74
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.write("Dropout Probability:", probability)
    if risk == "Low Risk":
        st.success(f"Risk Category: {risk}")
    elif risk == "Medium Risk":
        st.warning(f"Risk Category: {risk}")
    else:
        st.error(f"Risk Category: {risk}")