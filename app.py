from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

app = Flask(__name__)
model = load_model('mymodel.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the HTML form
        inputs = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['chestpain']),
            float(request.form['restingBP']),
            float(request.form['serumcholestrol']),
            float(request.form['fastingbloodsugar']),
            float(request.form['restingrelectro']),
            float(request.form['maxheartrate']),
            float(request.form['exerciseangia']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['noofmajorvessels'])
        ]

        # Print received inputs (for debugging)
        print("Received inputs:", inputs)

        # Prepare input as a 2D array (model expects a 2D array)
        input_array = np.array([inputs])

        # Predict using the model
        prediction = model.predict(input_array)

        # Print raw prediction (for debugging)
        print("Raw prediction:", prediction)

        # Convert output to a readable form (if prediction > 0.5, high risk, else low risk)
        result = 'High Risk' if prediction[0][0] > 0.5 else 'Low Risk'

        # Return the prediction as a JSON response
        return jsonify({'prediction': result})

    except Exception as e:
        # Handle any errors and return an error message
        print("Error during prediction:", str(e))
        return jsonify({'prediction': 'Error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
