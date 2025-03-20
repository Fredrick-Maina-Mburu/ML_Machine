from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow all origins

model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array([[
            int(data['Age']),
            int(data['Gender']),
            int(data['Polyuria']),
            int(data['Polydipsia']),
            int(data['sudden_weight_loss']),
        ]])
        prediction = model.predict([features])[0]
        return jsonify({'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
