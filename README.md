# ❤️ Diagnosify – AI-Powered Heart Disease Risk Assessment

Diagnosify is a machine learning-powered web application that provides an early assessment of an individual's risk of heart disease based on selected health and lifestyle information. It is designed as a decision-support tool to promote early awareness and encourage timely medical consultation.

> **Live Demo:** https://diagnosify-web-app-1.onrender.com/

---

## 📖 Overview

Cardiovascular diseases remain one of the leading causes of death worldwide. Early identification of potential risk factors can encourage preventive care and healthier lifestyle choices.

Diagnosify leverages a trained machine learning model to estimate the probability of heart disease using patient health metrics such as blood pressure, cholesterol, glucose level, BMI, and lifestyle habits.

**Disclaimer:** Diagnosify is intended for educational and screening purposes only. It does not provide medical diagnoses or replace professional healthcare advice.

---

## ✨ Features

* ❤️ Heart disease risk prediction
* 📊 Interactive risk visualization using a pie chart
* 📈 Heart Health Score
* ⚖️ Automatic BMI calculation and classification
* 🩺 Personalized explanation of factors influencing the assessment
* 📱 Responsive and user-friendly interface
* 🔒 AI screening disclaimer for responsible use

---

## 🛠️ Technologies Used

### Backend

* Flask
* Python

### Machine Learning

* XGBoost
* Scikit-learn
* Joblib
* NumPy
* Pandas

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

---

## 📋 Input Parameters

The model uses the following health indicators:

* Age
* Gender
* Height
* Weight
* Systolic Blood Pressure (ap_hi)
* Diastolic Blood Pressure (ap_lo)
* Cholesterol Level
* Glucose Level
* Smoking Status
* Alcohol Consumption
* Physical Activity

BMI is calculated automatically before prediction.

---

## 📊 Output

After prediction, Diagnosify provides:

* Estimated Heart Disease Risk
* Risk Category (Low, Moderate, or High)
* Heart Health Score
* BMI Value and BMI Classification
* Interactive Pie Chart
* Key Factors Influencing the Assessment
* Medical Disclaimer

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/Opemipo-001/Diagnosify-Web-App.git
```

### Navigate to the project folder

```bash
cd Diagnosify-Web-App
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python App.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```
Diagnosify-Web-App/
│
├── App.py
├── cardio_xgb_model.pkl
├── cardio_scaler.pkl
├── requirements.txt
├── templates/
├── static/
└── README.md
```

---

## 🎯 Project Objective

The objective of Diagnosify is to demonstrate the application of machine learning in healthcare by providing an accessible and user-friendly platform for heart disease risk assessment. The project combines predictive analytics with an intuitive web interface to showcase how artificial intelligence can support preventive healthcare.

---

## 👨‍💻 Author

**Opemipo Lawal**

* GitHub: https://github.com/Opemipo-001

---

## 📄 License

This project is intended for educational and research purposes.

It may be used, modified, and shared with appropriate attribution.
