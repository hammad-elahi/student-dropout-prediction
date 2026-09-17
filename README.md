# Student Dropout Prediction

A machine learning project that predicts whether a student is at risk of dropping out, using Logistic Regression. Built as part of the Big Brains Learning Machine Learning & Generative AI internship.

## Problem

Educational institutions often identify at-risk students too late, relying on manual observation rather than data. This project uses machine learning to predict dropout risk early, based on academic performance, financial status, and demographic data, enabling earlier intervention.

## Dataset

- **Source**: UCI Machine Learning Repository — [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (accessed via [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention))
- **Origin**: Data from a Portuguese higher education institution, covering multiple undergraduate programs
- **Records**: 4,424 students
- **Features**: 35 columns covering academic performance, demographics, and socioeconomic factors
- **Target**: Originally 3 classes (Dropout, Enrolled, Graduate), converted to binary classification (Dropout vs. Not Dropout) for this project

Note: some column names reflect the Portuguese academic system and differ from credit-hour/GPA systems used elsewhere. For example, "Curricular units approved" is roughly equivalent to "courses passed," and grades are on a 0-20 scale rather than a 4.0 GPA scale.

## Approach

1. **Data Cleaning**: Checked for missing values (none found), converted the target variable to binary
2. **Exploratory Data Analysis**: Analyzed relationships between features and dropout using box plots and correlation analysis
3. **Model Training**: Logistic Regression, trained on an 80/20 train-test split with MinMax feature scaling
4. **Evaluation**: Assessed using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
5. **Application**: Built an interactive Streamlit app for real-time risk prediction on new student data

## Key Findings

- Academic performance (curricular units approved per semester) and financial standing (tuition payment status) were the strongest predictors of dropout
- Demographic and macroeconomic factors (nationality, GDP, unemployment rate) showed little relationship with dropout
- The pattern of low academic performance correlating with dropout was consistent across both 1st and 2nd semesters

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85.9% |
| Precision (Dropout) | 87% |
| Recall (Dropout) | 71% |
| F1-Score (Dropout) | 78% |
| ROC-AUC | 0.91 |

## Live Application

[Link to be added after deployment]

## Files

- `student_dropout_prediction.py` - Full analysis pipeline (data cleaning, EDA, model training, evaluation)
- `app.py` - Streamlit web application for interactive risk prediction
- `model.pkl` - Trained Logistic Regression model
- `scaler.pkl` - Fitted MinMaxScaler
- `dataset.csv` - UCI Student Dropout dataset

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit

