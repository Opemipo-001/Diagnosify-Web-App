from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler.
xgb_model = joblib.load('cardio_xgb_model.pkl')
scaler = joblib.load('cardio_scaler.pkl')

# Feature order must match training exactly 
FEATURE_ORDER = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi'
]

# Validation ranges. These only reject physiologically
VALIDATION_RULES = {
    'age':         {'min': 1,   'max': 120,  'label': 'Age'},
    'height':      {'min': 50,  'max': 250,  'label': 'Height'},
    'weight':      {'min': 10,  'max': 300,  'label': 'Weight'},
    'ap_hi':       {'min': 50,  'max': 300,  'label': 'Systolic blood pressure'},
    'ap_lo':       {'min': 30,  'max': 200,  'label': 'Diastolic blood pressure'},
}


def validate_inputs(form):
    """
    Validate raw form inputs before they ever reach the model.
    Returns (cleaned_values_dict, list_of_errors).
    This is purely a safety net on the Flask side — it never
    changes what gets fed into scaler.transform().
    """
    errors = []
    values = {}

    required_fields = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active'
    ]

    # ---- Required / empty check ----
    for field in required_fields:
        raw = form.get(field, '').strip()
        if raw == '':
            errors.append(f"'{field}' is required.")

    if errors:
        return None, errors

    # ---- Type parsing ----
    try:
        values['age'] = float(form['age'])
        values['gender'] = int(form['gender'])
        values['height'] = float(form['height'])
        values['weight'] = float(form['weight'])
        values['ap_hi'] = float(form['ap_hi'])
        values['ap_lo'] = float(form['ap_lo'])
        values['cholesterol'] = int(form['cholesterol'])
        values['gluc'] = int(form['gluc'])
        values['smoke'] = int(form['smoke'])
        values['alco'] = int(form['alco'])
        values['active'] = int(form['active'])
    except ValueError:
        errors.append("One or more fields contain an invalid (non-numeric) value.")
        return None, errors

    # ---- Range checks ----
    for field, rule in VALIDATION_RULES.items():
        v = values[field]
        if v < rule['min'] or v > rule['max']:
            errors.append(
                f"{rule['label']} must be between {rule['min']} and {rule['max']}."
            )

    # ---- Logical check: systolic must exceed diastolic ----
    if values['ap_hi'] <= values['ap_lo']:
        errors.append("Systolic pressure (ap_hi) must be greater than diastolic pressure (ap_lo).")

    if errors:
        return None, errors

    return values, []


def get_risk_category(probability_pct):
    """
    Convert probability (%) into a risk category.
    0-30 = Low, 31-60 = Moderate, 61-100 = High.
    Purely a presentation-layer mapping on top of the model's
    own predict_proba() output — does not alter the prediction.
    """
    if probability_pct <= 30:
        return "Low Risk", "low"
    elif probability_pct <= 60:
        return "Moderate Risk", "moderate"
    else:
        return "High Risk", "high"


def get_bmi_category(bmi):
    """Standard WHO BMI categories — display only."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def get_health_score(probability_pct):
    """
    Heart Health Score / 100.
    Simple inverse of risk probability: a higher disease
    probability means a lower health score. Derived entirely
    from the model's predict_proba() output.
    """
    return round(100 - probability_pct)


def _evaluate_blood_pressure(ap_hi, ap_lo):
    """
    Returns (status, label, detail) for blood pressure.
    status: 'good' | 'caution' | 'risk'
    """
    reading = f"{int(ap_hi)}/{int(ap_lo)} mmHg"
    if ap_hi >= 140 or ap_lo >= 90:
        return ('risk',
                'Elevated blood pressure',
                f'Reading of {reading} falls in the hypertensive range, a major driver of cardiovascular risk.')
    elif ap_hi >= 130 or ap_lo >= 85:
        return ('caution',
                'Blood pressure is slightly elevated',
                f'Reading of {reading} is above the optimal range and worth monitoring.')
    else:
        return ('good',
                'Blood pressure is within a healthy range',
                f'Reading of {reading} is within normal limits.')


def _evaluate_cholesterol(cholesterol):
    if cholesterol == 3:
        return ('risk',
                'Cholesterol level significantly above normal',
                'Reported cholesterol is well above normal, which is strongly associated with increased cardiovascular risk.')
    elif cholesterol == 2:
        return ('caution',
                'Cholesterol is above the normal range',
                'Reported cholesterol is moderately elevated above normal.')
    else:
        return ('good',
                'Cholesterol level is normal',
                'Reported cholesterol falls within the normal range.')


def _evaluate_glucose(gluc):
    if gluc == 3:
        return ('risk',
                'Elevated glucose level',
                'Reported glucose is well above normal, a recognized cardiovascular risk contributor.')
    elif gluc == 2:
        return ('caution',
                'Glucose is above the normal range',
                'Reported glucose is moderately elevated above normal.')
    else:
        return ('good',
                'Glucose level is normal',
                'Reported glucose falls within the normal range.')


def _evaluate_bmi(bmi):
    if bmi >= 30:
        return ('risk',
                'BMI indicates obesity',
                f'BMI of {bmi:.1f} falls in the obese range, which increases strain on the cardiovascular system.')
    elif bmi >= 25:
        return ('caution',
                'BMI indicates overweight status',
                f'BMI of {bmi:.1f} falls in the overweight range.')
    elif bmi < 18.5:
        return ('caution',
                'BMI indicates underweight status',
                f'BMI of {bmi:.1f} falls below the healthy range.')
    else:
        return ('good',
                'BMI is within a healthy range',
                f'BMI of {bmi:.1f} falls within the normal range.')


def _evaluate_smoking(smoke):
    if smoke == 1:
        return ('risk',
                'Smoking increases cardiovascular risk',
                'Current smoking status is a significant contributor to heart disease risk.')
    else:
        return ('good', 'Non-smoker', 'No current smoking reported.')


def _evaluate_alcohol(alco):
    if alco == 1:
        return ('caution',
                'Alcohol consumption may contribute to increased risk',
                'Regular alcohol intake was reported.')
    else:
        return ('good', 'No reported alcohol consumption', 'No regular alcohol intake reported.')


def _evaluate_activity(active):
    if active == 0:
        return ('risk',
                'Lack of physical activity is associated with higher cardiovascular risk',
                'No regular physical activity was reported.')
    else:
        return ('good',
                'Physically active lifestyle reported',
                'Regular physical activity was reported, which supports cardiovascular health.')


def get_contributing_factors(values, bmi, risk_class):
    """
    Build "Key Factors Influencing This Assessment" dynamically
    from the patient's actual submitted values — never a static
    placeholder. Every dimension (BP, cholesterol, glucose, BMI,
    smoking, alcohol, activity) is evaluated against its own
    value and classified as good / caution / risk.

    Selection rule, tuned by overall risk_class so the tone of
    the section matches the result:
      - low:      show 'good' factors (what's working well),
                  plus any 'caution'/'risk' factors that exist
                  (a low-risk result can still carry one
                  individual flagged factor — show it honestly).
      - moderate: show a mix — both 'caution'/'risk' factors and
                  the 'good' factors, so the picture is balanced.
      - high:     lead with 'risk' and 'caution' factors (the
                  actual drivers); 'good' factors are included
                  only as secondary context.

    This never alters scaler.transform()/predict_proba()/predict()
    — it is purely a transparent, rule-based explanation layer
    built on top of the inputs already sent to the model.
    """
    evaluations = [
        _evaluate_blood_pressure(values['ap_hi'], values['ap_lo']),
        _evaluate_cholesterol(values['cholesterol']),
        _evaluate_glucose(values['gluc']),
        _evaluate_bmi(bmi),
        _evaluate_smoking(values['smoke']),
        _evaluate_alcohol(values['alco']),
        _evaluate_activity(values['active']),
    ]

    good = [{'label': l, 'detail': d, 'status': s} for (s, l, d) in evaluations if s == 'good']
    caution = [{'label': l, 'detail': d, 'status': s} for (s, l, d) in evaluations if s == 'caution']
    risk = [{'label': l, 'detail': d, 'status': s} for (s, l, d) in evaluations if s == 'risk']

    if risk_class == 'high':
        # Lead with the strongest signals, but still show positives if any exist
        ordered = risk + caution + good
    elif risk_class == 'moderate':
        # Balanced view: cautions/risks first, positives follow
        ordered = caution + risk + good
    else:
        # Low risk: lead with what's going well; still surface any
        # individual caution/risk factor truthfully if one exists
        ordered = good + caution + risk

    return ordered


@app.route('/')
def index():
    """Render the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Validate inputs, run the unchanged ML pipeline, build the dashboard view."""

    values, errors = validate_inputs(request.form)

    if errors:
        return render_template('index.html', errors=errors, form_data=request.form)

    age = values['age']
    gender = values['gender']
    height = values['height']
    weight = values['weight']
    ap_hi = values['ap_hi']
    ap_lo = values['ap_lo']
    cholesterol = values['cholesterol']
    gluc = values['gluc']
    smoke = values['smoke']
    alco = values['alco']
    active = values['active']

    #BMI calculation 
    bmi = weight / ((height / 100) ** 2)

    # Feature vector in the required order 
    features = np.array([[
        age, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active, bmi
    ]])

    # Scale + predict (UNCHANGED ML LOGIC) 
    scaled_features = scaler.transform(features)
    probability = xgb_model.predict_proba(scaled_features)[0][1]
    prediction = xgb_model.predict(scaled_features)[0]

    probability_pct = round(probability * 100, 2)
    safe_pct = round(100 - probability_pct, 2)

    risk_label, risk_class = get_risk_category(probability_pct)
    bmi_rounded = round(bmi, 1)
    bmi_category = get_bmi_category(bmi)
    health_score = get_health_score(probability_pct)
    contributing_factors = get_contributing_factors(values, bmi, risk_class)

    return render_template(
        'result.html',
        probability=probability_pct,
        safe_pct=safe_pct,
        risk_label=risk_label,
        risk_class=risk_class,
        prediction=int(prediction),
        bmi=bmi_rounded,
        bmi_category=bmi_category,
        health_score=health_score,
        contributing_factors=contributing_factors,
        inputs=values,
    )


if __name__ == '__main__':
    app.run(debug=True)