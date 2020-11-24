
from flask import Flask
from flask import request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
app = Flask(__name__)


@app.route('/prepare-data')
def getAndPrepareData():
    dataset = pd.read_csv('data/data.csv',encoding= 'unicode_escape')
    X = dataset.iloc[:, 0:4].values
    Y = dataset.iloc[:, 5:21].values

    lcpohlavi = LabelEncoder()
    X[:, 0] = lcpohlavi.fit_transform(X[:, 0])
    lcvzdelani = LabelEncoder()
    X[:, 2] = lcvzdelani.fit_transform(X[:, 2])
    lcobor = LabelEncoder()
    X[:, 3] = lcobor.fit_transform(X[:, 3])
    return X[1]


@app.route('/', methods=['GET'])
def home():
    year = request.args['year']
    return year