import pandas as pd

# Phase 2 - Dataset Exploration

df = pd.read_csv("dataset.csv")
print(df.shape)
print(df["Target"].value_counts())

# Phase 3 - Data Cleaning & Preprocessing

print(df.isnull().sum())
df["Target"] = df["Target"].map({"Dropout" : 1 , "Graduate" : 0 , "Enrolled" : 0})
print(df["Target"].value_counts())
print(df.head())
print(df.info())
print(df.describe())

# Phase 4 - Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Target" , data=df)
plt.title("Dropout Distribution")
plt.xlabel("Target (0 = Not Dropout, 1 = Dropout)")
plt.show()

sns.boxplot(x="Target" , y = "Curricular units 1st sem (approved)" , data=df)
plt.title("1st Semester Approved Units by Dropout Status")
plt.xlabel("Target (0 = Not Dropout, 1 = Dropout)")
plt.show()

sns.boxplot(x="Target" , y = "Curricular units 1st sem (grade)" , data=df)
plt.title("1st Semester Grade Status by Dropout Status")
plt.xlabel("Target (0 = Not Dropout, 1 = Dropout)")
plt.show()

sns.boxplot(x="Target" , y = "Curricular units 2nd sem (approved)" , data=df)
plt.title("2nd Semester Approved Units by Dropout Status")
plt.xlabel("Target (0 = Not Dropout, 1 = Dropout)")
plt.show()

sns.boxplot(x="Target" , y = "Curricular units 2nd sem (grade)" , data=df)
plt.title("2nd Semester Grade Status by Dropout Status")
plt.xlabel("Target (0 = Not Dropout, 1 = Dropout)")
plt.show()

plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.xticks(rotation=90 , fontsize = 8)
plt.yticks(rotation=0 , fontsize = 8)
plt.tight_layout()
plt.title('Feature Correlation Heatmap' , pad = 20)
plt.subplots_adjust(top=0.90, bottom=0.30)
plt.show()

correlations = df.corr()['Target'].sort_values(ascending=False)
print(correlations)

# Phase 5 - Model Training (Logistic Regression)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

X = df.drop("Target", axis = 1)
y = df["Target"]
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size= 0.2 , random_state= 42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled,y_train)
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)
print("\nPredictions: \n" , predictions[:5])
print("\nPrediction Probabilities: \n", probabilities[:5])

# Phase 6 - Model Evaluation

from sklearn.metrics import accuracy_score , confusion_matrix , classification_report, roc_auc_score

print("\nAccuracy Score: \n", accuracy_score(y_test , predictions))
print("\nConfusion Matrix: \n", confusion_matrix(y_test , predictions))
print("\nClassification Report: \n" ,classification_report(y_test , predictions))
print("\nROC AUC Score: ", roc_auc_score(y_test,  probabilities[:,1]))

# Phase 7 - Student Risk Prediciton

student_1 = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 4,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 5,
    "Father's occupation": 5,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 1,
    "Tuition fees up to date": 0,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 22,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 3,
    "Curricular units 1st sem (approved)": 1,
    "Curricular units 1st sem (grade)": 8.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 2,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (grade)": 0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74
}
student_1_df = pd.DataFrame([student_1])
print(student_1_df)

student_1_scaled = scaler.transform(student_1_df)
prediction = model.predict(student_1_scaled)
probability = model.predict_proba(student_1_scaled)
print("Prediction:", prediction)
print("Probability:", probability)

dropout_probability = probability[0][1]
if dropout_probability < 0.3:
    risk = "Low Risk"
elif dropout_probability < 0.7:
    risk = "Medium Risk"
else:
    risk = "High Risk"
    
print("Dropout Probability: ", dropout_probability)
print("Risk Category: ", risk)

student_2 = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 9,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 5,
    "Father's occupation": 5,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 0,
    "Scholarship holder": 1,
    "Age at enrollment": 19,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 14.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 6,
    "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 14.0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74
}

student_2_df = pd.DataFrame([student_2])
print(student_2_df)

student_2_scaled = scaler.transform(student_2_df)
prediction = model.predict(student_2_scaled)
probability = model.predict_proba(student_2_scaled)
print("Prediction:", prediction)
print("Probability:", probability)

dropout_probability = probability[0][1]
if dropout_probability < 0.3:
    risk = "Low Risk"
elif dropout_probability < 0.7:
    risk = "Medium Risk"
else:
    risk = "High Risk"
    
print("Dropout Probability: ", dropout_probability)
print("Risk Category: ", risk)

student_3 = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 11,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 5,
    "Father's occupation": 5,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 5,
    "Curricular units 1st sem (approved)": 2,
    "Curricular units 1st sem (grade)": 11.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 5,
    "Curricular units 2nd sem (approved)": 2,
    "Curricular units 2nd sem (grade)": 12.0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74
}
student_3_df = pd.DataFrame([student_3])
print(student_3_df)

student_3_scaled = scaler.transform(student_3_df)
prediction = model.predict(student_3_scaled)
probability = model.predict_proba(student_3_scaled)
print("Prediction:", prediction)
print("Probability:", probability)

dropout_probability = probability[0][1]
if dropout_probability < 0.3:
    risk = "Low Risk"
elif dropout_probability < 0.7:
    risk = "Medium Risk"
else:
    risk = "High Risk"
    
print("Dropout Probability: ", dropout_probability)
print("Risk Category: ", risk)

import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
