import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Setting the page configuration with a winter-themed icon
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="❄️")

# Add custom CSS for the winter theme
st.markdown("""
    <style>
    body {
        background-color: #f0f8ff;
        color: #000;
    }
    .stButton>button {
        background-color: #1e3d58;
        color: white;
        border-radius: 12px;
    }
    .stTextInput>label {
        color: #1e3d58;
        font-size: 18px;
    }
    .stTextInput>div>input {
        border: 2px solid #1e3d58;
        border-radius: 10px;
    }
    .stTitle {
        color: #1e3d58;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model
diabetes_model_path = "diabetes_model_sav"
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

# Title of the app with a wintery touch
st.title('❄️ Diabetes Prediction using ML ❄️')

# Create columns for user input with a winter theme
col1, col2, col3 = st.columns(3)

# Get user inputs
with col1:
    Pregnancies = st.text_input('❄️ Number of Pregnancies')
with col2:
    Glucose = st.text_input('❄️ Glucose Level')
with col3:
    BloodPressure = st.text_input('❄️ Blood Pressure value')

with col1:
    SkinThickness = st.text_input('❄️ Skin Thickness value')
with col2:
    Insulin = st.text_input('❄️ Insulin Level')
with col3:
    BMI = st.text_input('❄️ BMI value')

with col1:
    DiabetesPedigreeFunction = st.text_input('❄️ Diabetes Pedigree Function value')
with col2:
    Age = st.text_input('❄️ Age of the Person')

# Variable to store the diagnosis result
diab_diagnosis = ''

# When the user clicks the button
if st.button('❄️ Diabetes Test Result ❄️'):
    try:
        # Convert inputs to float and create the input array
        user_input = [
            float(Pregnancies), float(Glucose), float(BloodPressure), 
            float(SkinThickness), float(Insulin), float(BMI), 
            float(DiabetesPedigreeFunction), float(Age)
        ]

        # Make a prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Determine the diagnosis
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic.'
        else:
            diab_diagnosis = 'The person is not diabetic.'

    except ValueError:
        diab_diagnosis = 'Please enter valid numerical values for all fields.'

if st.button('❄️ Show Model Accuracy ❄️'):

    test_data = pd.read_csv(r"D:\workshop2\diabetes.csv")

    x_test = test_data.drop(columns=["outcome"])
    y_test = test_data["outcome"]

    y_pred = diabetes_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model accuracy on test data: {accuracy * 100:.2f}%")

    # Display the result
    st.success(diab_diagnosis)
