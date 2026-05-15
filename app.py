from flask import Flask, render_template, request

import pickle


# Load trained model
model = pickle.load(open('model.pkl', 'rb'))


# Load vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


# Create Flask app
app = Flask(__name__)


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get user message
    message = request.form['message']

    # Convert message into numbers
    data = vectorizer.transform([message])

    # Prediction
    prediction = model.predict(data)[0]

    # Convert output into readable text
    if prediction == 1:
        result = "SPAM MESSAGE"
    else:
        result = "NOT SPAM"

    return render_template('index.html', prediction=result)


# Run app
if __name__ == '__main__':
    app.run(debug=True)