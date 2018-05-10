###################################
# of course, use Keras Sequential
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
###################################
# collections - high performance datatypes
# so is numpy
import collections
import numpy as np
import os
import pyblackjack
###################################

# initialize parameters
DATA_DIR = ""
NUM_ACTIONS = 3 # number of valid actions (left, stay, right)
GAMMA = 0.99 # decay rate of past observations
INITIAL_EPSILON = 0.1 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
MEMORY_SIZE = 50000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 100
NUM_EPOCHS_TRAIN = 1000

BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

modelFile = "rl-network.h6"

# loads the model if model exists, otherwise we train it.
if os.path.exists(modelFile):
    model = load_model(os.path.join(DATA_DIR, modelFile))
else:
    # build the model
    # (3,
    #  4, -> (32 conv2d) -> (relu act'n) -> -> (64 conv2d) -> (relu act'n) -> (flatten) ->
    # (512 dense) -> (relu) -> (3 dense)
    #  1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, strides=1,
             kernel_initializer="normal",
             padding="same",
             input_shape=(3, 4, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=2, strides=1,
             kernel_initializer="normal",
             padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=2, strides=1,
             kernel_initializer="normal",
             padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="normal"))
    model.add(Activation("relu"))
    model.add(Dense(3, kernel_initializer="normal"))

model.compile(optimizer=Adam(lr=1e-6), loss="mse")

game = pyblackjack.Blackjack()
experience = collections.deque(maxlen=MEMORY_SIZE)

fout = open(os.path.join(DATA_DIR, "rl-simple-results.tsv"), "w")
num_games, num_wins = 0, 0
epsilon = INITIAL_EPSILON
for e in range(NUM_EPOCHS):
    loss = 0.0
    game.reset()

    # get first state
    a_0 = 0  # act = 0 (stand); 1 (hit); 2 (double)

    x_t, r_0 = game.oneStep(a_0)  # False, funds is the return values

    s_t = "observation"  # we'll make this the hands

