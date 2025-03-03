import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the saved model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Preprocess new data
def preprocess_data(data, scaler):
    # Ensure data is a DataFrame with correct feature names
    data_df = pd.DataFrame(data)
    # Scale the data
    data_scaled = scaler.transform(data_df)
    return data_scaled

# Make prediction
def predict(model, data):
    prediction = model.predict(data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return result

if __name__ == "__main__":
    # Define the path to the model and scaler
    model_path = 'diabetes_model.pkl'
    scaler_path = 'scaler.pkl'

    # Load the model and scaler
    model = load_model(model_path)

    # Example new data for prediction
    new_data = {
        'Pregnancies': [2],
        'Glucose': [120],
        'BloodPressure': [70],
        'SkinThickness': [20],
        'Insulin': [80],
        'BMI': [25.0],
        'DiabetesPedigreeFunction': [0.5],
        'Age': [30]
    }

    # Load and use the scaler
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Preprocess the new data
    new_data_scaled = preprocess_data(new_data, scaler)

    # Make prediction
    result = predict(model, new_data_scaled)

    print(f"Prediction for new data: {result}")
