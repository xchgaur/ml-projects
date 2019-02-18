from flask import Flask, request, session, g, url_for, render_template, jsonify
import logging
import json
import numpy
#import matplotlib.pyplot as plt
import pandas
import math
from fbprophet import Prophet

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.backend import clear_session
"""

import re

logger = logging.getLogger('tsanalyze')
app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello World"


@app.route('/getForecast', methods=['GET', 'POST'])
def process_details():
    error = None
    status_code = 500

    mystr = eval(request.form.get("json_data"))
    print(type(mystr))
    print(mystr)
    mydata = json.loads(mystr)
    print(mydata['column'])
    col = mydata['column'].strip()
    print("Requests data is {}".format(mydata))

    print(request.content_type)

    f = request.files['file']
    if f:
        print(f)
        f.save("temp.csv")
    else:
        logger.error("Could not fetch details from request.json/request.form")
        res = {'success': False, 'status_code': status_code, 'message': 'Could not retrieve file object from request'}
        return jsonify(res)

    df = pandas.read_csv('temp.csv')
    df.rename(columns={"time": 'ds', col: 'y'}, inplace=True)

    print(df.head())
    print(df.tail())

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=15)
    print(future.tail())
  
    forecast = m.predict(future)
    return jsonify(forecast)

if __name__ == '__main__':
   app.run(host='0.0.0.0')

