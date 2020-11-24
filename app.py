
from flask import Flask
from flask import request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
app = Flask(__name__)

@app.route('/get-data')
def getData():
    dataset = pd.read_csv('data/data.csv',encoding= 'unicode_escape')
    X = dataset.iloc[:, 0:4].values
    Y = dataset.iloc[:, 5:21].values
    return X


@app.route('/', methods=['GET'])
def home():
    year = request.args['year']
    return year