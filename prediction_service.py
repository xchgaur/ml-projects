from flask import Flask, request, session, g, url_for, render_template, jsonify
import logging

import numpy
#import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.backend import clear_session
import re

logger = logging.getLogger('mlapi')
app = Flask(__name__)


def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)


@app.route('/')
def hello_world():
   return "Hello World"


@app.route('/processReq', methods=['GET', 'POST'])
def process_details():
    error = None
    status_code = 500
    print(request.data)
    print(request.content_type)

    f = request.files['file']
    if f:
        print(f)
        f.save("temp.csv")
    else:
        logger.error("Could not fetch details from request.json/request.form")
        res = {'success': False, 'status_code': status_code, 'message': 'Could not retrieve file object from request'}
        return jsonify(res)


    # fix random seed for reproducibility
    numpy.random.seed(7)

    # load the dataset
    df = pandas.read_csv('temp.csv',index_col=0)
    print(df.head())
    print(df.tail())
    y = df.index
    y = y.astype('float32')

    next_slot = int(y[1] - y[0])
    temp = int(y[2] - y[1])
    temp2  = int(y[3] - y[2])

    if (next_slot != temp or next_slot != temp2):
        next_slot = 300 *int((next_slot + temp + temp2)/900)
    print("Next slot is {}".format(next_slot))
     

    
    last_unix = df.index.max()
    print("Last dat is {}".format(last_unix))

    dataframe = pandas.read_csv('temp.csv', usecols=[1], engine='python')
    print(dataframe.head())
    dataset = dataframe.values
    dataset = dataset.astype('float32')


    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    #train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train, test = dataset, dataset[train_size:len(dataset),:]
    print("Datast length is {} , Train len is {}, test length is {} ".format(len(dataset), len(train), len(test)))

    #train_size = int(len(dataset))
    #test_size = len(dataset) - forecast_out

    #train, test = dataset[0:train_size,:], dataset[-test_size:len(dataset),:]
    #print(len(train), len(test))

    #trial = numpy.array(test)


    # reshape into X=t and Y=t+1
    look_back = int(70 * 1800.0/next_slot)
    print("Look back is {}".format(look_back))

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)


    print("Trainx len is {}, TrainY length is {}, testX length is {} , testY length is {}".format(len(trainX), len(trainY), len(testX), len(testY)))

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    #trial = numpy.reshape(trial, (trial.shape[0], 1, trial.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    #trialPredict = model.predict(trial)


    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))


    #print(trainY[0][-500:])
    #print("\n#############################################\n")
    #print(trainPredict[-500:])

    #print("\n#############################################\n")
    #print("\n#############################################\n")
    #print(testY[0][-50:])
    #print("\n#############################################\n")
    #print(testPredict[-50:])

    show = [(x,y) for x,y in zip(testY[0][-50:], testPredict[-50:])]
    for val in show:
        print(val)


    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\tGoing to predict future values\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    ref = dataset[len(dataset)-look_back:, 0]

    #next_slot = 60 * 30
    next_unix = last_unix + next_slot

    insight_predictions = []
    num_of_predictions = int(350 * 1800.0/next_slot)
    for i in range(num_of_predictions):
        sample = []
        sample.append(ref)
        trial = numpy.array(sample)
        trial = numpy.reshape(trial, (trial.shape[0], 1, trial.shape[1]))
        trialPredict = model.predict(trial)
        trialPredictVal = scaler.inverse_transform(trialPredict)
        print("\tValues predicted is\t{}".format(trialPredictVal))

        curr_arr = []
        curr_arr.append(str(next_unix))
        curr_arr.append(str(trialPredictVal))
        insight_predictions.append(curr_arr)

        temp = ref[1:]
        ref = numpy.append(temp,trialPredict[0,0]) 
        next_unix = next_unix + next_slot
    clear_session()
    status_code = 200
    res = {'success': True, 'status_code': status_code, 'message': 'Successfully provided predictions', 'values': insight_predictions}
    return jsonify(res)    

if __name__ == '__main__':
   app.run(host='0.0.0.0')

