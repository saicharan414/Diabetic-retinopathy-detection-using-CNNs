from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(f)) for f in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)