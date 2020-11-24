
from flask import Flask
from flask import request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def return_prediction(model, scaler):
    pred = model.predict(scaler.transform(np.array([[
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 55
    ]])))

    return pred

app = Flask(__name__)

# Loading the model
my_model = load_model("/model/final_fieldpredictor_model.h5")
my_scaler = joblib.load("/model/myScaler.pkl")


@app.route('/prediciton')
def prediction():
    results = return_prediction(my_model, my_scaler)
    return results