import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

def train_model(file_path="used_device_data.csv"):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found. Please provide the dataset file to train the model.")
        
    # Load the data
    df = pd.read_csv(file_path)
    
    # Data preprocessing
    # Replace yes/no with 1/0
    df.replace({'yes': 1, 'no': 0}, inplace=True)
    
    # Drop irrelevant columns if they exist
    if 'model' in df.columns:
        df.drop(columns=['model'], inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Encode categorical columns using one-hot encoding
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Define features and target
    X = df.drop('normalized_used_price', axis=1)
    y = df['normalized_used_price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values
    X_train.fillna(X_train.mean(numeric_only=True), inplace=True)
    X_test.fillna(X_test.mean(numeric_only=True), inplace=True)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model columns for future reference
    model_columns = list(X_train.columns)
    
    # Save the model and columns
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(model_columns, f)
    
    return model, model_columns

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        return model, model_columns
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def predict_price(input_data, model, model_columns):
    """Predict the price based on input data"""
    if model is None or model_columns is None:
        raise ValueError("Model not loaded. Please ensure the model is trained before making predictions.")
        
    # Create a DataFrame with the correct columns
    input_df = pd.DataFrame([input_data])
    
    # Handle categorical variables
    input_df.replace({'yes': 1, 'no': 0}, inplace=True)
    
    # Create a DataFrame with all model columns set to 0
    prediction_df = pd.DataFrame(columns=model_columns)
    prediction_df.loc[0] = [0] * len(model_columns)
    
    # Fill in the values we have from input_data
    for column in input_df.columns:
        if column in prediction_df.columns:
            prediction_df[column] = input_df[column].values
    
    # For device_brand and os, create the appropriate dummy columns
    if 'device_brand' in input_data:
        brand_col = f"device_brand_{input_data['device_brand']}"
        if brand_col in prediction_df.columns:
            prediction_df[brand_col] = 1
    
    if 'os' in input_data:
        os_col = f"os_{input_data['os']}"
        if os_col in prediction_df.columns:
            prediction_df[os_col] = 1
    
    # Make prediction
    prediction = model.predict(prediction_df)[0]
    return prediction

if __name__ == "__main__":
    # Train and save the model if it doesn't exist
    model, model_columns = load_model()
    if model is None:
        try:
            print("Training model...")
            model, model_columns = train_model()
            print("Model trained and saved successfully!")
        except Exception as e:
            print(f"Error training model: {str(e)}")
    else:
        print("Model loaded successfully!") 