from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
with open('models/diabetes_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# Load the dataset to extract feature names
data = pd.read_csv('diabetes.csv')
feature_columns = ['Pregnancies', 'Glucose ( normal:70-99mg/dl)', 'BloodPressure ( normal:120/80 mm Hg)', 'SkinThickness', 'Insulin (30-78 pmol/L)', 'BMI', 'DiabetesPedigreeFunction ', 'Age']

@app.route('/')
def index():
    return render_template('index.html', features=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(request.form[feature]) for feature in feature_columns]
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    prediction = model.predict(user_input)

    if prediction[0] == 1:
        result = "The model predicts that the user has diabetes."
    else:
        result = "The model predicts that the user does not have diabetes."
    
    return render_template('index.html', features=feature_columns, result=result)

if __name__ == '__main__':
    app.run(debug=True)

