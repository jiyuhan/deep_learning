import numpy as np
import requests
import random
import math
import pandas_datareader.data as web
from datetime import datetime
import os
import pickle

X_train = []
Y_train = []
X_test = []
Y_test = []


def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out


# load_purpose: training data = 0, 300 days game = 1
# 300 trading days ago is: 2017 February 20 (Monday)
def loadStock(stock, window_size, predict_days, load_purpose=0):
    """
    Return the historical stock market close prices. Is done with a web API call.
    stock = 'SPY' or 'QQQ' for this assignment.
    """
    if load_purpose == 0:
        filepath = stock + '.pkl'
        start = datetime(2013, 3, 25)  # 3/25/2013 is Monday
        end = datetime(2018, 3, 28)
    elif load_purpose == 1:
        filepath = stock + '_GAME' + '.pkl'
        start = datetime(2017, 2, 20) # 300 days
        end = datetime(2018, 4, 16)

    if not os.path.exists(filepath):
        # get data from morningstar API
        data = web.DataReader(stock, 'morningstar', start, end)
        data = data.values
        print("Loading successful: data.shape =", data.shape)
        values = data[:, 0]  # get stock's closed values
        f = open(filepath, 'wb')
        pickle.dump(values, f)
        f.close()
        # print("File", filepath, ".pkl is created; if you change the data, remove this file first.")
    else:
        f = open(filepath, 'rb')
        values = pickle.load(f)

    # print("closed values.shape =", values.shape)
    X = []
    # it basically has X be like this:
    # [a, b, c, d, f]
    # [   b, c, d, f, g]
    # ...
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
    # from 5 to end
    Y = X[predict_days:]
    # from 0 to len - 5
    X = X[:-predict_days]

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    print("X.shape =", X.shape, "Y.shape =", Y.shape)
    return X, Y


def generate_x_y_data_v5(isTest, batch_size, predict_days, load_purpose=0):
    """
    Return financial data for the stock symbol SPY.

    For every window (i.e, seq_length), Y is the prediction following X.
    Train and test data are separated according to the 90/10 rule.
    Every example in X contains seq_length points of SPY data 
    in the feature axis/dimension.
    It is to be noted that the returned X and Y has the same shape
    and are in a tuple.
    """
    # print("Standard & Poors 500 Index ETF Trust")
    # step_length is the number for encoder and decoder's backpropagation.
    # A small number is used for demo.
    seq_length = 60
    global Y_train
    global X_train
    global X_test
    global Y_test
    global x_spy
    global y_spy

    if predict_days > seq_length:
        predict_days = seq_length

    if len(Y_test) == 0 or load_purpose == 1:
        x_spy, y_spy = loadStock(
            stock='SPY', window_size=seq_length, predict_days=predict_days, load_purpose=load_purpose)

        # Split 90-10:  X (and Y) is a list of vectors (of length seq_length)
        m = int(len(x_spy) * 0.9)
        X_train = x_spy[: m]
        Y_train = y_spy[: m]
        X_test = x_spy[m:]
        Y_test = y_spy[m:]

    if isTest == 0:
        # return a random set of batch_size items from (X_train, Y_train)
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    if isTest == 1:
        # return a random set of batch_size items from (X_test, Y_test)
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)
    if load_purpose == 1:
        X_out = np.array(x_spy[-batch_size:]).transpose((1, 0, 2))
        Y_out = np.array(y_spy[-batch_size:]).transpose((1, 0, 2))
        return X_out, Y_out
    # return the last batch_size items in (X_test, Y_test)
    X_out = np.array(X_test[-batch_size:]).transpose((1, 0, 2))
    Y_out = np.array(Y_test[-batch_size:]).transpose((1, 0, 2))
    return X_out, Y_out
