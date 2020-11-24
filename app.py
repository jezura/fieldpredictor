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

    number=pred[0][0]

    return number



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
my_model = load_model("final_fieldpredictor_model.h5")
my_scaler = joblib.load("myScaler.pkl")




@app.route('/prediction')
def prediction():
    results = return_prediction(model=my_model, scaler=my_scaler)
    return str(results)


if __name__ == '__main__':
    app.run(debug=True)