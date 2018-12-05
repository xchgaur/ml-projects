from flask import Flask, request, session, g, url_for, render_template, jsonify
import logging
from keras.models import load_model
from keras.backend import clear_session

#loaded = load_model('am_detect.h5')
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import re




logger = logging.getLogger('mlapi')
app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello World"


@app.route('/processReq', methods=['GET', 'POST'])
def process_details():
    error = None
    status_code = 500
    print(request.data)
    print(request.content_type)

    req_obj = None

    f = request.files['file']
    if f:
        print(f)
    elif request.form:
        req_obj = request.form
    elif request.json:
        req_obj =  request.json
    else:
        logger.error("Could not fetch details from request.json/request.form")
        res = {'success': False, 'status_code': status_code, 'message': 'Unsupported request type.Please contact support team.'}
        return jsonify(res)

    loaded = load_model('my_model.h5')
    df = pd.read_csv(f,index_col=0)
    print(df.head())

    X_train, X_test = train_test_split(df, test_size=0)
    y_train = X_train['Label']
    X_train = X_train.drop(['Label'], axis=1)
    print(len(X_train))
    print(len(y_train))
    
    X_train = X_train.values
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    predictions = loaded.predict(X_train_scaled)

    mse = np.mean(np.power(X_train_scaled - predictions, 2), axis=1)
    df_error = pd.DataFrame({'reconstruction_error': mse, 'Label': y_train}, index=y_train.index)
    print(len(df_error))
    ret = str(df_error.describe())
    print(ret)
    input = ret.strip().split("\n")[3]
    std_deviation = float(re.sub(r'\s+', " ", input).split(" ")[-1])
    print(std_deviation)

    sigma_level = 15
    threshold = sigma_level * std_deviation

    outliers = df_error.index[df_error.reconstruction_error > threshold].tolist()
    print(outliers)
    out_data = ""
    for val in outliers:
        print(df.loc[val])
        out_data = out_data + "\n\{" + str(val) + ": " + str(df.loc[val]) + "\}" 

    clear_session()
    del loaded 
    #logger.debug("Request details ---------------- \n{}".format(req_obj))
    #print("Request details ---------------- \n{}".format(req_obj))
    ret_msg = {}
    ret_msg.update({'status' : 'pass', 'msg' : out_data})
    return jsonify(ret_msg)

if __name__ == '__main__':
   app.run(host='0.0.0.0')
