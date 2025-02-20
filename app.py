import pickle
import streamlit as st

# Setting the page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Load the pre-trained model
diabetes_model_path = r"D:\workshop2\diabetes_model_sav"
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))


# Title of the app
st.title('Diabetes Prediction using ML')

# Create columns for user input
col1, col2, col3 = st.columns(3)

# Get user inputs
with col1:
    Pregnancies = st.text_input('Number of Pregnancies')
with col2:
    Glucose = st.text_input('Glucose Level')
with col3:
    BloodPressure = st.text_input('Blood Pressure value')

with col1:
    SkinThickness = st.text_input('Skin Thickness value')
with col2:
    Insulin = st.text_input('Insulin Level')
with col3:
    BMI = st.text_input('BMI value')

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
with col2:
    Age = st.text_input('Age of the Person')


# Variable to store the diagnosis result
diab_diagnosis = ''

# When the user clicks the button
if st.button('Diabetes Test Result'):
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
if st.button('show model accuracy'):

    test_data = pd.read_csv(r"D:\workshop2\diabetes.csv")

    x_test = test_data.drop(cpoumns=["outcome"])
    y_test = test_data["outcome"]

    y_pred = diabetes_model.predict(x_test)


    # Display the result
    st.success(diab_diagnosis)



