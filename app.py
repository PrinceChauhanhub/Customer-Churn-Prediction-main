## Gender -> 0 Female  1 Male
## Churn -> 0 No  1 Yes
## Scaler is exported as scaler.pkl
## Model is exported as model.pickle
## Order of X is :['Age', 'Gender', 'Tenure', 'MonthlyCharges']


import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()
st.write("Enter the values and hit the predict button for getting prediction.")

st.divider()

age = st.number_input("Enter_age",min_value = 10, max_value = 100, value = 30)

gender = st.selectbox("Enter the Gender",["Male","Female"])

tenure = st.number_input("enter Tenure",min_value = 0, max_value = 130, value = 10)

monthly_charges = st.number_input("Enter Monthly Charges",min_value = 30, max_value = 150)

st.divider()

predictbutton = st.button("Predict")

st.divider()

if predictbutton:
    gender_selected = 0 if gender == "Female" else 1
    X = [age, gender_selected, tenure, monthly_charges ]
    
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    
    prediction = model.predict(X_array)[0]
    
    predicted = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
    
    st.balloons()
    st.write(f"Predicted: {predicted}")
    
else:
    st.write("Please enter the values and hit predict button")
    
