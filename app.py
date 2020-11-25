from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from flask import request
from wtforms import TextField, SubmitField
from wtforms.validators import NumberRange

import numpy as np
from tensorflow.keras.models import load_model
import joblib


def make_predictions(model, scaler, gender, age, edu_lvl, edu_field):
    # data and codes

    edu_lvl_codes = [
        [0, 0, 0, 0, 0, 0, 1],  # Zakladni vzdelani
        [0, 0, 0, 0, 0, 1, 0],  # Vyuceni nebo uplne stredni bez maturity
        [1, 0, 0, 0, 0, 0, 0],  # Stredni s maturitou
        [0, 0, 0, 0, 1, 0, 0],  # Vyssi odborne
        [0, 1, 0, 0, 0, 0, 0],  # Vysokoskolske bakalarske
        [0, 0, 0, 1, 0, 0, 0],  # Vysokoskolske magisterske (inzenyrske)
        [0, 0, 1, 0, 0, 0, 0]  # Vysokoskolske doktorske
    ]

    if edu_lvl == 0:
        edu_lvl_code = edu_lvl_codes[0]
    elif edu_lvl == 1:
        edu_lvl_code = edu_lvl_codes[1]
    elif edu_lvl == 2:
        edu_lvl_code = edu_lvl_codes[2]
    elif edu_lvl == 3:
        edu_lvl_code = edu_lvl_codes[3]
    elif edu_lvl == 4:
        edu_lvl_code = edu_lvl_codes[4]
    elif edu_lvl == 5:
        edu_lvl_code = edu_lvl_codes[5]
    elif edu_lvl == 6:
        edu_lvl_code = edu_lvl_codes[6]
    else:
        edu_lvl_code = [0]

    np_arr2 = np.array(edu_lvl_code)

    edu_field_codes = [
        # Vseobecny obor (vzdelani) - (Zakladni nebo Gymnazium)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        # IT a management
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Obchod a ekonomie
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Pedagogika, ucitelstvi a telovychova
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Teologie
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # Filosofie, politologie, historie, psychologie, sociologie, verejna sprava, soc. obory
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Doprava a logistika
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Pravo
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Prirodni vedy - chemie, fyzika, ekologie, matematika,..
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # Lingvistika, jazykove skoly
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # !!!!!!!!!!!OPRAVIT, NEBYLA V DATECH!!!!!!!!!!!!
        # Elektrotechnika, mechanika, technika, prumysl (vcetne napr. automechaniku apod.)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Zdravotnictvi, medicina, veterina
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        # Stavebnictvi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Zemedelstvi, lesnictvi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        # Obrana a ochrana
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Umeni, umelecke skoly (hudba, film, konzervatore, fotograf, grafika, umel. kovar, ..)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        # Sluzby, hotelnictvi, gastronomie, cest. ruch (prodavac, kuchar, krejci, kadernice, ..)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Ostatni nezaraditelna remesla (kovar, tesar, zamecnik, truhlar, ..)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]

    if edu_field == 0:
        edu_field_code = edu_field_codes[0]
    elif edu_field == 1:
        edu_field_code = edu_field_codes[1]
    elif edu_field == 2:
        edu_field_code = edu_field_codes[2]
    elif edu_field == 3:
        edu_field_code = edu_field_codes[3]
    elif edu_field == 4:
        edu_field_code = edu_field_codes[4]
    elif edu_field == 5:
        edu_field_code = edu_field_codes[5]
    elif edu_field == 6:
        edu_field_code = edu_field_codes[6]
    elif edu_field == 7:
        edu_field_code = edu_field_codes[7]
    elif edu_field == 8:
        edu_field_code = edu_field_codes[8]
    elif edu_field == 9:
        edu_field_code = edu_field_codes[9]
    elif edu_field == 10:
        edu_field_code = edu_field_codes[10]
    elif edu_field == 11:
        edu_field_code = edu_field_codes[11]
    elif edu_field == 12:
        edu_field_code = edu_field_codes[12]
    elif edu_field == 13:
        edu_field_code = edu_field_codes[13]
    elif edu_field == 14:
        edu_field_code = edu_field_codes[14]
    elif edu_field == 15:
        edu_field_code = edu_field_codes[15]
    elif edu_field == 16:
        edu_field_code = edu_field_codes[16]
    elif edu_field == 17:
        edu_field_code = edu_field_codes[17]
    else:
        edu_field_code = [0]

    np_arr1 = np.array(edu_field_code)
    np_arr3 = np.array([gender])
    np_arr4 = np.array([age])

    arr = np.concatenate((np_arr1, np_arr2, np_arr3, np_arr4))
    final_array_list = list(arr)

    pred = model.predict(scaler.transform(np.array([final_array_list])))
    pred = pred[0]

    return pred


app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'

# REMEMBER TO LOAD THE MODEL AND THE SCALER!
my_model = load_model("final_fieldpredictor_model.h5")
my_scaler = joblib.load("myScaler.pkl")


@app.route('/get-jobfields-relevance-scores', methods=['GET'])
def prediction():
    gender = int(request.args['gender'])
    age = int(request.args['age'])
    edu_lvl = int(request.args['edu_lvl'])
    edu_field = int(request.args['edu_field'])

    results = make_predictions(model=my_model, scaler=my_scaler,
                               gender=gender, age=age, edu_lvl=edu_lvl, edu_field=edu_field)
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
