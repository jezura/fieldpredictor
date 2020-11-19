
from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    year = request.args['year']
    return year