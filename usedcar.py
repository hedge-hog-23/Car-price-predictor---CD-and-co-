import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

# Load your dataset as a DataFrame
data = pd.read_csv('cardekho_dataset.csv')  # Replace 'your_dataset.csv' with your actual file path

# Create a copy of the DataFrame
data_copy = data.copy()

# Select relevant columns
data_copy = data_copy[['brand', 'model', 'vehicle_age', 'km_driven', 'fuel_type', 'transmission_type', 'selling_price']]

# Initialize LabelEncoders
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()
fuel_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

# Fit and transform the training data
data_copy['brand'] = brand_encoder.fit_transform(data_copy['brand'])
data_copy['model'] = model_encoder.fit_transform(data_copy['model'])
data_copy['fuel_type'] = fuel_type_encoder.fit_transform(data_copy['fuel_type'])
data_copy['transmission_type'] = transmission_encoder.fit_transform(data_copy['transmission_type'])

# Split the data into features and target
X = data_copy.drop('selling_price', axis=1)
y = data_copy['selling_price']

# Train a simple model (you should use your own trained model)
model = RandomForestRegressor()
model.fit(X, y)

# Save the model and encoders for future use
joblib.dump(model, 'your_model.joblib')
joblib.dump(brand_encoder, 'brand_encoder.joblib')
joblib.dump(model_encoder, 'model_encoder.joblib')
joblib.dump(fuel_type_encoder, 'fuel_type_encoder.joblib')
joblib.dump(transmission_encoder, 'transmission_encoder.joblib')

def predict_price(user_brand, user_model, user_age, user_km_driven, user_fuel_type, user_transmission):
    # Load the saved encoders
    brand_encoder = joblib.load('brand_encoder.joblib')
    model_encoder = joblib.load('model_encoder.joblib')
    fuel_type_encoder = joblib.load('fuel_type_encoder.joblib')
    transmission_encoder = joblib.load('transmission_encoder.joblib')

    # Preprocess user input
    user_brand_transformed = brand_encoder.transform([user_brand])[0]
    user_model_transformed = model_encoder.transform([user_model])[0]
    user_fuel_type_transformed = fuel_type_encoder.transform([user_fuel_type])[0]
    user_transmission_transformed = transmission_encoder.transform([user_transmission])[0]

    # Make prediction
    input_data = pd.DataFrame({'brand': [user_brand_transformed],
                               'model': [user_model_transformed],
                               'vehicle_age': [user_age],
                               'km_driven': [user_km_driven],
                               'fuel_type': [user_fuel_type_transformed],
                               'transmission_type': [user_transmission_transformed]})

    predicted_price = model.predict(input_data)[0]
    return predicted_price

# Streamlit GUI
st.image('final.jpg', caption='Competitor : Das and co', use_column_width=True)
st.sidebar.header("User Input")

# Dropdowns for user input
user_brand = st.sidebar.selectbox("Select car brand", data['brand'].unique())
# Filter models based on selected brand
filtered_models = data[data['brand'] == user_brand]['model'].unique()

# Dropdown for user input - Model
user_model = st.sidebar.selectbox("Select car model", filtered_models)
user_age = st.sidebar.number_input("Enter vehicle age", min_value=0.0, max_value=100.0, value=1.0)
user_km_driven = st.sidebar.number_input("Enter kilometers driven", min_value=0.0, value=10000.0)
user_fuel_type = st.sidebar.selectbox("Select fuel type", data['fuel_type'].unique())
user_transmission = st.sidebar.selectbox("Select transmission type", data['transmission_type'].unique())

# Predict button
if st.sidebar.button("Predict"):
    predicted_price = predict_price(user_brand, user_model, user_age, user_km_driven, user_fuel_type, user_transmission)
    st.success(f"Predicted Price: Rupees {np.round(predicted_price, 2)}")
