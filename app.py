from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    # Convert numeric fields
    numeric_fields = [
        "screen_size", "rear_camera_mp", "front_camera_mp",
        "ram", "battery", "weight", "release_year",
        "days_used", "original_price"
    ]

    for field in numeric_fields:
        data[field] = float(data[field])

    # Convert Yes/No to 1/0
    data["has_4g"] = 1 if data["has_4g"] == "Yes" else 0
    data["has_5g"] = 1 if data["has_5g"] == "Yes" else 0

    # Create normalized_new_price
    data["normalized_new_price"] = data["original_price"] / 20000

    # Create DataFrame
    df = pd.DataFrame([data])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Model prediction
    pred_used = model.predict(df)[0]

    # Get normalized_new_price
    norm_new = df["normalized_new_price"].values[0]

    # Convert to ratio
    ratio = pred_used / norm_new if norm_new != 0 else 0.5
    ratio = max(0.3, min(ratio, 0.9))

    # 🔥 FINAL PRICE LOGIC 
    real_price = ratio * data["original_price"]

    age = 2026 - data["release_year"]
    usage_days = data["days_used"]

    age_factor = max(0.6, 1 - (age * 0.08))
    usage_factor = max(0.7, 1 - (usage_days / 2000))

    real_price = real_price * age_factor * usage_factor

    # Final safety
    real_price = min(real_price, data["original_price"])

    real_price = int(real_price)

    return render_template(
        "result.html",
        price_in_rupees=real_price,
        input_data=data
    )


if __name__ == "__main__":
    app.run(debug=True)