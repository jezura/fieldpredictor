from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model, scaler):
    pred = model.predict(scaler.transform(np.array([[
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 55
    ]])))

    pred=pred[0]

    return pred



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
my_model = load_model("final_fieldpredictor_model.h5")
my_scaler = joblib.load("myScaler.pkl")




@app.route('/get-jobfields-relevance-scores')
def prediction():
    results = return_prediction(model=my_model, scaler=my_scaler)
    return (str(results[0]) + '<br>'
            + str(results[1]) + '<br>'
            + str(results[2]) + '<br>'
            + str(results[3]) + '<br>'
            + str(results[4]) + '<br>'
            + str(results[5]) + '<br>'
            + str(results[6]) + '<br>'
            + str(results[7]) + '<br>'
            + str(results[8]) + '<br>'
            + str(results[9]) + '<br>'
            + str(results[10]) + '<br>'
            + str(results[11]) + '<br>'
            + str(results[12]) + '<br>'
            + str(results[13]) + '<br>'
            + str(results[14]) + '<br>'
            + str(results[15]) + '<br>')


if __name__ == '__main__':
    app.run(debug=True)