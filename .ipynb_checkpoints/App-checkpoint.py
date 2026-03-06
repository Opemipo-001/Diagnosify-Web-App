from flask import Flask, render_template, request
import numpy as np
import joblib

# Load trained model
model = joblib.load("diagnosify_lr_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease Detected"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)