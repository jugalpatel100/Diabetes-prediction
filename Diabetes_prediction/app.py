from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_values = [float(x) for x in request.form.values()]
    final_features = [np.array(feature_values)]
    prediction_prob = model.predict_proba(final_features)

    prediction_text = ""

    '''if prediction == 0:
        prediction_text = "You do not have diabetes."
    else:
        prediction_text = "You have diabetes."
    '''

    if prediction_prob[0][1] < 0.3:
        prediction_text = "You do not have diabetes."
    elif prediction_prob[0][1] < 0.6:
        prediction_text = "You have a low chance of having diabetes."
    elif prediction_prob[0][1] < 0.8:
        prediction_text = "You hava a considerable chance of having diabetes."
    else:
        prediction_text = "You have diabetes."

    return render_template('index.html', prediction_outcome=prediction_text)



if __name__ == "__main__":
    app.run(debug=True)