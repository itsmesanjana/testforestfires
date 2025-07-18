from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the model and scaler
ridge_model = pickle.load(open('models/ridge_model.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Create DataFrame
        input_data = pd.DataFrame([[temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                                  columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])

        # Scale the input
        scaled_input = scaler_model.transform(input_data)

        # Predict
        result = ridge_model.predict(scaled_input)[0]

        # Return prediction to template
        return render_template('home.html', result=round(result, 2))

    else:
        return render_template('home.html', result=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
