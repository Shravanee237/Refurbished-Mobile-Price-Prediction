# Mobile Device Price Predictor

A Flask web application that predicts the normalized used price of mobile devices based on various specifications.

## Features

- Predicts mobile device used prices based on a trained Linear Regression model
- Intuitive web interface for entering device specifications
- RESTful API endpoint for making predictions programmatically
- Responsive design

## Requirements

- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-Learn

## Setup and Installation

1. Clone this repository or download the project files.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your `used_device_data.csv` file in the project root directory.

   **Note:** If you don't have the dataset file, the application will run in demo mode but won't be able to make predictions.

## Dataset Structure

The application expects a CSV file named `used_device_data.csv` with the following columns:
- device_brand: Brand of the mobile device
- os: Operating system
- screen_size: Size of the screen in inches
- 4g: Whether the device has 4G connectivity (yes/no)
- 5g: Whether the device has 5G connectivity (yes/no)
- rear_camera_mp: Rear camera megapixels
- front_camera_mp: Front camera megapixels
- internal_memory: Internal storage in GB
- ram: RAM size in GB
- battery: Battery capacity in mAh
- weight: Weight in grams
- release_year: Year the device was released
- days_used: Number of days the device has been used
- normalized_new_price: Normalized new price of the device
- normalized_used_price: Normalized used price of the device (target variable)

## Running the Application

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`

3. Enter the device specifications in the form and click "Predict Price" to get the prediction.

## API Usage

You can also use the API endpoint to make predictions programmatically:

```python
import requests
import json

# API endpoint
url = 'http://127.0.0.1:5000/api/predict'

# Device specifications
data = {
    'device_brand': 'Apple',
    'os': 'iOS',
    'screen_size': 6.1,
    '4g': 'yes',
    '5g': 'yes',
    'rear_camera_mp': 12.0,
    'front_camera_mp': 12.0,
    'internal_memory': 128.0,
    'ram': 6.0,
    'battery': 3240.0,
    'weight': 174.0,
    'release_year': 2022,
    'days_used': 180,
    'normalized_new_price': 5.8
}

# Make prediction request
response = requests.post(url, json=data)
result = response.json()

# Print prediction
print(f"Predicted normalized used price: {result['prediction']}")
``` 