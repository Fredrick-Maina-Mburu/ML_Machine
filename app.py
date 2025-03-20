from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.json
    try:
        features = [
            int(data['Age']),
            int(data['Gender']),
            int(data['Polyuria']),
            int(data['Polydipsia']),
            int(data['sudden_weight_loss']),
            
        ]
        prediction = model.predict([features])[0]
        return jsonify({'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
