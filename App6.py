from flask import Flask, render_template, request
import joblib 
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('personality_model1.joblib')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = request.form.to_dict()

    # Assuming the form has input fields corresponding to personality traits
    input_features = [float(input_data['Openness']),
                      float(input_data['Conscientiousness']),
                      float(input_data['Extraversion']),
                      float(input_data['Agreeableness']),
                      float(input_data['Neuroticism'])]

    # Make a prediction using the trained model
    prediction = model.predict([input_features])

    # Render the result page with the prediction
    return render_template('result.html', prediction=prediction[0])

# Run the app
if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5000,debug=False)
