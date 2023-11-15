from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

THIS_FOLDER = Path(__file__).parent.resolve()
saved_model = THIS_FOLDER / "model.pickle"

model = pickle.load(open(saved_model, 'rb'))
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']

# home page
@app.route('/', methods=['GET','POST'])
def home():
    return render_template("home.html")

# prediction results page
@app.route('/predict_result', methods=['POST'])
def predict():
    int_features = np.array([x for x in request.form.values()])
    data_unseen = pd.DataFrame([int_features], columns = cols)
    prediction = int(model.predict(data_unseen))
    prediction_proba = model.predict_proba(data_unseen)
    return render_template('results.html',pred='{} ({:.2f})'.format(prediction, prediction_proba[0][1]),
                           len=len(cols), columns=cols, vals = int_features)


# endpoint api for using the prediction engine
@app.route('/predict_api', methods=['POST'])
def predict_api():
    json_ = request.json
    query = pd.DataFrame(json_)
    predict = list(model.predict(query))
    return jsonify({'prediction': str(predict)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
