# car_price_predictor.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

class CarPricePredictor:
    def __init__(self):
        # Loading dataset "cardheko.csv"
        self.data = pd.read_csv('cardekho_dataset.csv')

        self.data_copy = self.data.copy()

        # Select relevant columns
        self.data_copy = self.data_copy[['brand', 'model', 'vehicle_age', 'km_driven', 'fuel_type', 'transmission_type', 'selling_price']]

        # Initialize LabelEncoders
        self.brand_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()
        self.fuel_type_encoder = LabelEncoder()
        self.transmission_encoder = LabelEncoder()

        # Fit and transform the training data
        self.data_copy['brand'] = self.brand_encoder.fit_transform(self.data_copy['brand'])
        self.data_copy['model'] = self.model_encoder.fit_transform(self.data_copy['model'])
        self.data_copy['fuel_type'] = self.fuel_type_encoder.fit_transform(self.data_copy['fuel_type'])
        self.data_copy['transmission_type'] = self.transmission_encoder.fit_transform(self.data_copy['transmission_type'])

        # Split the data into features and target
        self.X = self.data_copy.drop('selling_price', axis=1)
        self.y = self.data_copy['selling_price']

        # Train the model
        self.model = RandomForestRegressor()
        self.model.fit(self.X, self.y)

        # Save the model and encoders for future use
        joblib.dump(self.model, 'your_model.joblib')
        joblib.dump(self.brand_encoder, 'brand_encoder.joblib')
        joblib.dump(self.model_encoder, 'model_encoder.joblib')
        joblib.dump(self.fuel_type_encoder, 'fuel_type_encoder.joblib')
        joblib.dump(self.transmission_encoder, 'transmission_encoder.joblib')

    def predict_price(self, user_brand, user_model, user_age, user_km_driven, user_fuel_type, user_transmission):
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

        predicted_price = self.model.predict(input_data)[0]
        return predicted_price
