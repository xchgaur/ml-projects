import warnings
import numpy
import pandas
#import matplotlib
#import seaborn
#import plotly
import tensorflow
import keras
import pickle
import re

print('Numpy version      :' , numpy.__version__)
print('Pandas version     :' ,pandas.__version__)
#print('Matplotlib version :' ,matplotlib.__version__)
#print('Seaborn version    :' , seaborn.__version__)
#print('Plotly version     :', plotly.__version__)
print('Tensorflow version :' , tensorflow.__version__)
print('Keras version      :' , keras.__version__)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
#import matplotlib.pyplot as plt
#plt.rcdefaults()
#from pylab import rcParams
#import seaborn as sns

import datetime
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
#from ann_visualizer.visualize import ann_viz
# 
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_auc_score, auc,
                             precision_score, recall_score, roc_curve, precision_recall_curve,
                             precision_recall_fscore_support, f1_score,
                             precision_recall_fscore_support)


df = pd.read_csv('final_stats.csv',index_col=0)
#df = df.drop(columns='time')
print(df.index)
print(df.head())
print(df.shape)
#print(df.loc[df.index])

print(df.Label.value_counts())
print(df.Label.value_counts(normalize=True)*100)
print(df.columns)

numerical_cols = ['memory', 'cpu', 'disk', 'inbw', 'outBw', 'Label']
print(df.Label.unique())

labels = df['Label'].astype(int)
labels[labels != 0] = 1
print(len(labels[labels !=0]))
print(df.Label.value_counts().tolist())
print(df.Label.astype(str).unique().tolist())

RANDOM_SEED = 101

X_train, X_test = train_test_split(df, test_size=0.2, random_state = RANDOM_SEED)

X_train = X_train[X_train['Label'] == 0]
X_train = X_train.drop(['Label'], axis=1)
y_test  = X_test['Label']
X_test  = X_test.drop(['Label'], axis=1)
X_train = X_train.values
X_test  = X_test.values
print('Training data size   :', X_train.shape)
print('Validation data size :', X_test.shape)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

input_dim = X_train.shape[1]
encoding_dim = 4

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(3), activation="tanh")(encoder)
encoder = Dense(int(2), activation="tanh")(encoder)
decoder = Dense(int(3), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

def train_validation_loss(df_history):
    
    trace = []
    
    for label, loss in zip(['Train', 'Validation'], ['loss', 'val_loss']):
        trace0 = {'type' : 'scatter', 
                  'x'    : df_history.index.tolist(),
                  'y'    : df_history[loss].tolist(),
                  'name' : label,
                  'mode' : 'lines'
                  }
        trace.append(trace0)
    data = Data(trace)
    
    layout = {'title' : 'Model train-vs-validation loss', 'titlefont':{'size' : 30},
              'xaxis' : {'title':  '<b> Epochs', 'titlefont':{ 'size' : 25}},
              'yaxis' : {'title':  '<b> Loss', 'titlefont':{ 'size' : 25}},
              }


nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer='adam', loss='mse' )

t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train_scaled, X_train_scaled,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0
                        )

t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))



df_history = pd.DataFrame(history.history)

autoencoder.save('my_model.h5')
predictions = autoencoder.predict(X_test_scaled)

mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'Label': y_test}, index=y_test.index)
ret = df_error.describe()
print(type(ret))

print(ret)
input = str(ret).strip().split("\n")[3]
std_deviation = float(re.sub(r'\s+', " ", input).split(" ")[-1])

sigma_level = 15
threshold = sigma_level * std_deviation

data_n = pd.DataFrame(X_test_scaled, index= y_test.index, columns=numerical_cols[0:-1])
def compute_error_per_dim(point):
    
    initial_pt = np.array(data_n.loc[point,:]).reshape(1,5)
    reconstrcuted_pt = autoencoder.predict(initial_pt)
    
    return abs(np.array(initial_pt  - reconstrcuted_pt)[0])

outliers = df_error.index[df_error.reconstruction_error > threshold].tolist()
for val in outliers:
    print(df.loc[val])  
print(outliers)
print(len(outliers))
#file1 = open("ae_obj",'wb')
#pickle.dump(autoencoder,file1)


RE_per_dim = {}
for ind in outliers:
    RE_per_dim[ind] = compute_error_per_dim(ind)
    
RE_per_dim = pd.DataFrame(RE_per_dim, index= numerical_cols[:-1]).T
print(RE_per_dim.head())

#print(df_error.loc[1387])
#print(data_n.loc[1387])

#print(autoencoder.predict(np.array(data_n.loc[1387]).reshape(1,4)))

