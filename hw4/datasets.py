
import numpy as np
import requests

import random
import math

__author__ = "Guillaume Chevalier"
__license__ = "MIT"
__version__ = "2017-03"


def generate_x_y_data_v1(isTrain, batch_size, predict_days):
    """
    Data for exercise 1.

    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offset
    commonly applied to X an Y.

    The returned arrays are of shape:
        (seq_length, batch_size, output_dim)
        Therefore: (10, batch_size, 2)

    For this exercise, let's ignore the "isTrain"
    argument and test on the same data.
    """
    seq_length = 10
    if (predict_days > seq_length):
        predict_days = seq_length

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        x1 = sig1[:seq_length]
        y1 = sig1[predict_days:predict_days+seq_length]
        x2 = sig2[:seq_length]
        y2 = sig2[predict_days:predict_days+seq_length]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y


def generate_x_y_data_two_freqs(isTrain, batch_size, seq_length, predict_days):
    if (predict_days > seq_length):
        predict_days = seq_length
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        )

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sig1 = amp_rand * np.cos(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        ) + sig1

        x1 = sig1[:seq_length]
        y1 = sig1[predict_days:predict_days+seq_length]

        x_ = np.array([x1])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y


def generate_x_y_data_v2(isTrain, batch_size, predict_days):
    """
    Similar the the "v1" function, but here we generate a signal with
    2 frequencies chosen randomly - and this for the 2 signals. Plus,
    the lenght of the examples is of 15 rather than 10.
    So we have 30 total values for past and future.
    """
    return generate_x_y_data_two_freqs(
        isTrain, batch_size, seq_length=15, predict_days=predict_days)


def generate_x_y_data_v3(isTrain, batch_size, predict_days):
    """
    Similar to the "v2" function, but here we generate a signal
    with noise in the X values. Plus,
    the lenght of the examples is of 30 rather than 10.
    So we have 60 total values for past and future.
    """
    seq_length = 30
    x, y = generate_x_y_data_two_freqs(
        isTrain, batch_size, seq_length=seq_length, predict_days=predict_days)
    noise_amount = random.random() * 0.15 + 0.10
    x = x + noise_amount * np.random.randn(seq_length, batch_size, 1)

    avg = np.average(x)
    std = np.std(x) + 0.0001
    x = x - avg
    y = y - avg
    x = x / std / 2.5
    y = y / std / 2.5

    return x, y


def loadCurrency(curr, window_size, predict_days):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2018-03-26&currency={}".format(
            curr
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]
    print("kept_values length =", len(kept_values))

    X = []
    for i in range(len(kept_values) - window_size):
        X.append(kept_values[i:i + window_size])
    Y = X[predict_days:]
    X = X[:-predict_days]

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    print("X.shape =", X.shape, "Y.shape =", Y.shape)
    return X, Y


def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X


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


X_train = []
Y_train = []
X_test = []
Y_test = []


def generate_x_y_data_v4(isTest, batch_size, predict_days):
    """Return financial data for the bitcoin.

    Features are USD and EUR, in the internal dimension.
    We normalize X and Y data according to the X only to not
    spoil the predictions we ask for.

    For every window (window or seq_length), Y is the prediction
    following X.  Train and test data are separated according to the
    90/10 rule.  Every example in X contains 40 points of USD and then
    EUR data in the feature axis/dimension.  It is to be noted that
    the returned X and Y has the same shape and are in a tuple.

    """
    # 40 step_num for encoder and decoder's backpropagation.
    seq_length = 40
    if (predict_days > seq_length):
        predict_days = seq_length

    global Y_train
    global X_train
    global X_test
    global Y_test

    # First load, with memoization:
    if len(Y_test) == 0:
        print("Bitcoin price prediction")
        # API call:
        X_usd, Y_usd = loadCurrency(
            "USD", window_size=seq_length, predict_days=predict_days)
        X_eur, Y_eur = loadCurrency(
            "EUR", window_size=seq_length, predict_days=predict_days)

        # All data, aligned:
        X = np.concatenate((X_usd, X_eur), axis=2)
        Y = np.concatenate((Y_usd, Y_eur), axis=2)
        X, Y = normalize(X, Y)  # X.shape == Y.shape

        # Split 90-10:  X (and Y) is a list of vectors (of length seq_length)
        m = int(len(X) * 0.9)
        X_train = X[: m]
        Y_train = Y[: m]
        X_test = X[m:]
        Y_test = Y[m:]

    if isTest == 0:
        # return a random set of batch_size items from (X_train, Y_train)
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    elif isTest == 1:
        # return a random set of batch_size items from (X_test, Y_test)
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)
    else:
        # return the last batch_size items in (X_test, Y_test)
        X_out = np.array(X_test[-batch_size:]).transpose((1, 0, 2))
        Y_out = np.array(Y_test[-batch_size:]).transpose((1, 0, 2))
        return X_out, Y_out


# Get stoack quotes

import pandas_datareader.data as web
from datetime import datetime


def loadStock(stock, window_size, predict_days):
    """
    Return the historical stock market close prices. Is done with a web API call.
    stock = 'SPY' or 'QQQ' for this assignment.
    """
    filepath = stock + '.pkl'
    if not os.path.exists(filepath):
        # start = datetime(2008, 3, 31) # 3/31/2008 is Monday
        start = datetime(2013, 3, 25)  # 3/25/2013 is Monday
        end = datetime(2018, 3, 28)
        # get data from morningstar API
        data = web.DataReader(stock, 'morningstar', start, end)
        data = data.values
        print("Loading successful: data.shape =", data.shape)
        # data[:, 0] = Close value
        # data[:, 1] = High value
        # data[:, 2] = Low value
        # data[:, 3] = Open value
        # data[:, 4] = Volume; if Volume = 0, the market is closed that day
        values = data[:, 0]  # get stock's closed values
        f = open(filepath, 'wb')
        pickle.dump(values, f)
        f.close()
        print("File", filepath,
              ".pkl is created; if you change the data, remove this file first.")
    else:
        f = open(filepath, 'rb')
        values = pickle.load(f)

    print("closed values.shape =", values.shape)
    X = []
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
    Y = X[predict_days:]
    X = X[:-predict_days]

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    print("X.shape =", X.shape, "Y.shape =", Y.shape)
    return X, Y


def generate_x_y_data_v5(isTest, batch_size, predict_days):
    """
    Return financial data for the stock symbol SPY.

    For every window (i.e, seq_length), Y is the prediction following X.
    Train and test data are separated according to the 90/10 rule.
    Every example in X contains seq_length points of SPY data 
    in the feature axis/dimension.
    It is to be noted that the returned X and Y has the same shape
    and are in a tuple.
    """
    # step_length is the number for encoder and decoder's backpropagation.
    # A small number is used for demo.
    seq_length = 60
    if (predict_days > seq_length):
        predict_days = seq_length

    # to be completed
