import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the pre-trained model
model = joblib.load('diabetes_model.joblib')

# Define the feature names as used during model training
feature_names = [
    'Encoded Gender', 'age', 'hypertension', 'heart_disease',
    'Encoded smoking_history', 'bmi', 'blood_glucose_level', 'diabetes'
]

# Define a function to make predictions
def predict_diabetes(input_data):
    # Convert the input data to a DataFrame with the correct feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Make a prediction
    prediction = model.predict(input_df)[0]
    return prediction

# Dictionaries for encoding and decoding
smoking_history_dict = {
    "No Info": 0.0,
    "current": 1.0,
    "ever": 2.0,
    "former": 3.0,
    "never": 4.0,
    "not current": 5.0
}

gender_dict = {
    "Female": 0.0,
    "Male": 1.0,
    "Other": 2.0
}

boolean_dict = {
    "Yes": 1.0,
    "No": 0.0,
}

# Inverse dictionaries for decoding
inverse_gender_dict = {v: k for k, v in gender_dict.items()}
inverse_smoking_history_dict = {v: k for k, v in smoking_history_dict.items()}
inverse_boolean_dict = {v: k for k, v in boolean_dict.items()}

def fun_color_map(prediction):
    if prediction is not None:
        if prediction <= 6.0:
            adherence_color, adherence_text = "#2ecc71", "Excellent Adherence"
        elif 6.0 < prediction < 8.0:
            adherence_color, adherence_text = "#27ae60", "Good Adherence"
        elif 8.0 <= prediction < 10.0:
            adherence_color, adherence_text = "yellow", "Fair Adherence"
        else:
            adherence_color, adherence_text = "red", "Poor Adherence"
    else:
        # Default values if prediction is not available
        adherence_color, adherence_text = "black", "No Prediction Yet"
    
    return adherence_color, adherence_text

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Medicine Adherence Prediction</h1><br><br>", unsafe_allow_html=True)

# Initialize the session state if it doesn't exist
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'submitted_data_human_readable' not in st.session_state:
    st.session_state.submitted_data_human_readable = []

# Create two main columns with custom width ratio
main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    # Create two sub-columns for input fields
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
        gender_encoded = gender_dict[gender]
        
        age = st.number_input('Age', min_value=0.0, max_value=120.0, value=25.0)

        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
        hypertension_encoded = boolean_dict[hypertension]

        heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
        heart_disease_encoded = boolean_dict[heart_disease]

    with sub_col2:
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=40.0, max_value=800.0, value=140.0)

        smoking_history = st.selectbox('Smoking History', list(smoking_history_dict.keys()))
        smoking_history_encoded = smoking_history_dict[smoking_history]

        bmi = st.number_input('BMI', min_value=10.0, max_value=100.0, value=25.0)
        
        diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
        diabetes_encoded = boolean_dict[diabetes]

    input_data = {
        'Encoded Gender': gender_encoded,
        'age': age,
        'hypertension': hypertension_encoded,
        'heart_disease': heart_disease_encoded,
        'Encoded smoking_history': smoking_history_encoded,
        'bmi': bmi,
        'blood_glucose_level': blood_glucose_level,
        'diabetes': diabetes_encoded
    }

    # Predict button
    if st.button('Predict'):
        # Make the prediction
        prediction = predict_diabetes(input_data)

        # Get color and text for adherence
        adherence_color, adherence_text = fun_color_map(prediction)

        # Append the prediction and input data to the session state
        st.session_state.predictions.append({
            'input_data': input_data,
            'prediction': prediction,
            'adherence': adherence_text
        })
        st.session_state.submitted_data_human_readable.append({
            'Gender': gender,
            'Age': age,
            'Hypertension': hypertension,
            'Heart Disease': heart_disease,
            'Smoking History': smoking_history,
            'BMI': bmi,
            'Glucose Level': blood_glucose_level,
            'Diabetes': diabetes,
            'Predicted HBA1C': prediction,
            'Adherence': adherence_text
        })

        st.markdown(f'Prediction: The predicted HBA1C Level is: <span style="color:{adherence_color};">{prediction}</span>, <span style="color:{adherence_color};">{adherence_text}</span>', unsafe_allow_html=True)

prediction = None  # Define prediction variable outside the button blocks

with main_col2:
    if 'submitted_data_human_readable' in st.session_state and st.session_state.submitted_data_human_readable:
        df_human_readable = pd.DataFrame(st.session_state.submitted_data_human_readable)

        # JavaScript for custom cell styles
        colorAdherence_js = JsCode("""
        function(params) {
            let color = 'black';
            if (params.value == 'Excellent Adherence') {
                color = '#2ecc71';
            } else if (params.value == 'Good Adherence') {
                color = '#27ae60';
            } else if (params.value == 'Fair Adherence') {
                color = 'yellow';
            } else if (params.value == 'Poor Adherence') {
                color = 'red';
            }
            return { 'color': color };
        }
        """)

        colorHBA1C_js = JsCode("""
        function(params) {
            let color = 'black';
            if (params.value <= 6.0) {
                color = '#2ecc71';
            } else if (params.value > 6.0 && params.value < 8.0) {
                color = '#27ae60';
            } else if (params.value >= 8.0 && params.value < 10.0) {
                color = 'yellow';
            } else if (params.value >= 10.0) {
                color = 'red';
            }
            return { 'color': color };
        }
        """)

        # Build AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_human_readable)
        gb.configure_pagination()
        gb.configure_default_column(editable=False, groupable=True, flex = 1, minWidth=80, maxWidth=500, resizable=True,)
        gb.configure_column("Adherence", cellStyle=colorAdherence_js)
        gb.configure_column("Predicted HBA1C", cellStyle=colorHBA1C_js)
        gb.configure_grid_options(domLayout='normal')

        gridOptions = gb.build()
         

        # Display the AgGrid table
        st.write('Submitted Data:')
        AgGrid(df_human_readable, gridOptions=gridOptions ,enable_enterprise_modules=False, allow_unsafe_jscode=True, theme='streamlit')

        # Optionally, provide a button to clear the data
        if st.button('Clear Data'):
            st.session_state.predictions = []
            st.session_state.submitted_data_human_readable = []
            st.experimental_rerun()
