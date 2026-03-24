import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Define features and target
feature_columns = ['Pregnancies', 'Glucose( normal:70-99mg/dl)', 'BloodPressure( normal:120/80 mm Hg)', 'SkinThickness', 'Insulin(30-78 pmol/L)', 'BMI', 'DiabetesPedigreeFunction(0.08-2.42)', 'Age']
target_column = 'Outcome'

X = data[feature_columns]
y = data[target_column]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and scaler
with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)
