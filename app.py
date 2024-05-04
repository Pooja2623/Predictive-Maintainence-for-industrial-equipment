import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

model = joblib.load('xgb_model.pkl')
failure_label_mapping = {
    1: 'No Failure',
    3: 'Power Failure',
    5: 'Tool Wear Failure',
    2: 'Overstrain Failure',
    4: 'Random Failures',
    0: 'Heat Dissipation Failure'
}

type_label_mapping = {
    'M': 2,
    'L': 1,
    'H': 0
}

# Function to preprocess input data and make predictions
def predict_failure(UDI, Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear, Type_Encoded):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'UDI': [UDI],
        'Air temperature': [Air_temperature],
        'Process temperature': [Process_temperature],
        'Rotational speed': [Rotational_speed],
        'Torque': [Torque],
        'Tool wear': [Tool_wear],
        'Type_Encoded': [Type_Encoded]
    })

    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    prediction = failure_label_mapping[prediction_encoded]
    return prediction

# Streamlit UI
st.title('Predict Machine Failure')
st.write('Enter the following parameters to predict machine failure:')

# Input fields
UDI = st.number_input('UDI', step=1)
Air_temperature = st.number_input('Air temperature', format="%.1f")
Process_temperature = st.number_input('Process temperature', format="%.1f")
Rotational_speed = st.number_input('Rotational speed', format="%.1f")
Torque = st.number_input('Torque', format="%.1f")
Tool_wear = st.number_input('Tool wear', format="%.1f")
Type_Input = st.selectbox('Type', list(type_label_mapping.keys()))

# Convert input type to encoded value
Type_Encoded = type_label_mapping.get(Type_Input)

# Prediction button
if st.button('Predict'):
    if Type_Encoded is not None:
        # Call the prediction function
        prediction = predict_failure(UDI, Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear, Type_Encoded)
        
        # Display prediction
        st.write('Prediction:', prediction)
    else:
        st.error("Invalid type selected. Please select a valid type.")
