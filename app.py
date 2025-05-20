from flask import Flask, request, render_template, jsonify
import pandas as pd
import model as model_utils
import os
import locale

app = Flask(__name__)

# Set locale for number formatting
locale.setlocale(locale.LC_ALL, '')

# Custom filter for formatting numbers with commas
@app.template_filter('format_number')
def format_number(value):
    try:
        return format(int(value), ',d')
    except (ValueError, TypeError):
        try:
            # For floating point numbers
            whole, fract = str(value).split('.')
            return f"{format(int(whole), ',d')}.{fract}"
        except:
            return value

# Check if dataset exists
csv_path = "used_device_data.csv"
dataset_exists = os.path.exists(csv_path)

# Default values if file not found
device_brands = ["Apple", "Samsung", "Xiaomi", "Oppo", "Vivo", "Huawei", "Nokia", "OnePlus", "Realme", "Motorola"]
operating_systems = ["Android", "iOS", "Windows", "Others"]

# Conversion factor from normalized price to rupees (assuming 1 normalized unit = 15000 rupees)
# This can be adjusted based on your data normalization scheme
PRICE_CONVERSION_FACTOR = 15000

# Only attempt to load the model if dataset exists
if dataset_exists:
    # Load the model at startup
    model, model_columns = model_utils.load_model()
    if model is None:
        model, model_columns = model_utils.train_model()
    
    # Get list of device brands and OS from data
    try:
        df = pd.read_csv(csv_path)
        device_brands = sorted(df['device_brand'].unique())
        operating_systems = sorted(df['os'].unique())
    except Exception as e:
        print(f"Warning: Unable to load brand and OS data: {str(e)}")
else:
    model = None
    model_columns = None
    print(f"Warning: Dataset file '{csv_path}' not found. App will run in demo mode only.")

@app.route('/')
def home():
    return render_template('index.html', 
                          device_brands=device_brands,
                          operating_systems=operating_systems)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return render_template('error.html', error_message=f"Model not available. Dataset file '{csv_path}' not found.")
        
        # Get input values from the form
        input_data = {
            'device_brand': request.form['device_brand'],
            'os': request.form['os'],
            'screen_size': float(request.form['screen_size']),
            '4g': request.form['4g'],
            '5g': request.form['5g'],
            'rear_camera_mp': float(request.form['rear_camera_mp']),
            'front_camera_mp': float(request.form['front_camera_mp']),
            'internal_memory': float(request.form['internal_memory']),
            'ram': float(request.form['ram']),
            'battery': float(request.form['battery']),
            'weight': float(request.form['weight']),
            'release_year': int(request.form['release_year']),
            'days_used': int(request.form['days_used']),
            'normalized_new_price': float(request.form['normalized_new_price'])
        }
        
        # Make prediction
        normalized_price = model_utils.predict_price(input_data, model, model_columns)
        
        # Convert normalized price to rupees
        price_in_rupees = round(normalized_price * PRICE_CONVERSION_FACTOR, 2)
        normalized_price_rounded = round(normalized_price, 2)
        
        # Return the prediction
        return render_template('result.html', 
                              prediction=normalized_price_rounded,
                              price_in_rupees=price_in_rupees,
                              input_data=input_data)
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'message': f"Model not available. Dataset file '{csv_path}' not found."
            }), 400
            
        # Get input data from JSON
        data = request.get_json()
        
        # Make prediction
        normalized_price = model_utils.predict_price(data, model, model_columns)
        
        # Convert normalized price to rupees
        price_in_rupees = normalized_price * PRICE_CONVERSION_FACTOR
        
        # Return prediction as JSON
        return jsonify({
            'prediction_normalized': normalized_price,
            'prediction_rupees': price_in_rupees,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 