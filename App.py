from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
xgb_model = joblib.load("cardio_xgb_model.pkl")
scaler = joblib.load("cardio_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    # Default prediction values
    prediction_msg = ""
    prediction_color = ""
    probability = 0

    # Store previous input values to keep form filled
    input_data = {
        "age": "",
        "gender": "",
        "height": "",
        "weight": "",
        "ap_hi": "",
        "ap_lo": "",
        "cholesterol": "",
        "gluc": "",
        "smoke": "",
        "alco": "",
        "active": ""
    }

    if request.method == "POST":
        if "predict" in request.form:
            # Get input values
            for key in input_data.keys():
                input_data[key] = request.form.get(key)

            # Convert to numeric types
            sample_patient = pd.DataFrame({
                "age": [int(input_data['age'])],
                "gender": [int(input_data['gender'])],
                "height": [float(input_data['height'])],
                "weight": [float(input_data['weight'])],
                "ap_hi": [int(input_data['ap_hi'])],
                "ap_lo": [int(input_data['ap_lo'])],
                "cholesterol": [int(input_data['cholesterol'])],
                "gluc": [int(input_data['gluc'])],
                "smoke": [int(input_data['smoke'])],
                "alco": [int(input_data['alco'])],
                "active": [int(input_data['active'])]
            })

            # Compute BMI exactly like training
            sample_patient['bmi'] = sample_patient['weight'] / (sample_patient['height']/100)**2

            # Ensure proper column order
            feature_order = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                             'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
            sample_patient = sample_patient[feature_order]

            # Scale features
            sample_scaled = scaler.transform(sample_patient)

            # Predict probability and class
            prob = xgb_model.predict_proba(sample_scaled)[0][1]  # Probability of heart disease
            pred = xgb_model.predict(sample_scaled)[0]

            probability = int(prob * 100)  # For gauge display

            # Set message and color
            if pred == 1:
                prediction_msg = "❌ High risk of heart disease"
                prediction_color = "red"
            else:
                prediction_msg = "✅ Low risk of heart disease"
                prediction_color = "green"

        elif "clear" in request.form:
            # Clear all inputs
            input_data = {key: "" for key in input_data.keys()}
            prediction_msg = ""
            prediction_color = ""
            probability = 0

    return render_template("index.html",
                           input_data=input_data,
                           prediction_msg=prediction_msg,
                           prediction_color=prediction_color,
                           probability=probability)


if __name__ == "__main__":
    app.run(debug=True)