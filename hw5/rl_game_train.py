# -*- coding: utf-8 -*-

# using Keras tl train a reinforcement learning model so it can play black-jack.

# HOW TO TRAIN?
# it seems like you are supposed to let the play plays with itself.
# WITH WHAT DATA?
# at winning, has reward, otherwise ends the game.

# after the model is trained and saved in a h6 file.
# play the game

# stand
# hit
# double

from __future__ import division, print_function
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
###################################
# import the paddle_game.py file
import paddle_game


"""
This function reshapes a frame into an understandable form.
"""
def preprocess_frames(frames):
    # frames.shape[0] is the number of frames in the "frames"
    # if it's less than 4, it means there's only one frame
    if frames.shape[0] < 4:
        # if it's a single frame, then
        # print(frames)
        # loads the first frame (even if there are multiple, only read the first)
        x_t = frames[0].astype("float") # enforce strict type binding to float (casting)
        x_t /= 80.0 # dunno why
        # this makes the data becomes from 4 * (shape x * shape y) to (4 * shape x * shape y)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=1)
        # s_t.shape = (3, 4), duplicate x_t 4 times.
    
    # if it's not less than 4, it means it contains 4 frames by design
    else:
        # print(frames)
        # 4 frames
        xt_list = []
        for i in range(4): # frames.shape[0]):
            x_t = frames[i].astype("float")
            x_t /= 80.0
            xt_list.append(x_t)
        s_t = np.stack((xt_list[0], xt_list[1], xt_list[2], xt_list[3]), axis=1)
    s_t = np.expand_dims(s_t, axis=2)
    s_t = np.expand_dims(s_t, axis=0)
    # s_t.shape = (1, 3, 4, 1)
    return s_t


"""
This function requests a batch and returns the batch
"""
def get_next_batch(experience, model, num_actions, gamma, batch_size):
    batch_indices = np.random.randint(low=0, high=len(experience),
                                      size=batch_size)
    batch = [experience[i] for i in batch_indices]
    #batch = experience[batch_indices]
    # batch is a list of experiences: s_t, a_t, r_t, s_tp1, game_over
    X = np.zeros((batch_size, 3, 4, 1)) 
    # X is a batch_size of frames.
    Y = np.zeros((batch_size, num_actions)) 
    # Y is a batch size of rewards (for each action).
    for i in range(len(batch)):
        s_t, a_t, r_t, s_tp1, game_over = batch[i]
        X[i] = s_t
        Y[i] = model.predict(s_t)[0]
        Q_sa = np.max(model.predict(s_tp1)[0])
        if game_over:
            Y[i, a_t] = r_t
        else:
            Y[i, a_t] = r_t + gamma * Q_sa
    return X, Y

if __name__ == "__main__":   
############################# main ###############################

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
        #  4, -> (32 conv2d) -> (relu act'n) -> -> (64 conv2d) -> (relu act'n) -> (flatten) -> (512 dense) -> (relu) -> (3 dense)
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

    # train network
    game = paddle_game.PaddleGame()
    experience = collections.deque(maxlen=MEMORY_SIZE)


    
    fout = open(os.path.join(DATA_DIR, "rl-simple-results.tsv"), "w")
    num_games, num_wins = 0, 0
    epsilon = INITIAL_EPSILON
    for e in range(NUM_EPOCHS):
        loss = 0.0
        game.reset()
    
        # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, r_0, game_over = game.step(a_0) 
        s_t = preprocess_frames(x_t)

        while not game_over:
            s_tm1 = s_t
            # next action
            if e <= NUM_EPOCHS_OBSERVE or np.random.rand() <= epsilon:
                a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
            else:
                q = model.predict(s_t)[0]
                a_t = np.argmax(q)
                
            # apply action, get reward
            x_t, r_t, game_over = game.step(a_t)
            s_t = preprocess_frames(x_t)
            # if reward, increment num_wins
            if r_t == 1: num_wins += 1
            # store experience
            experience.append((s_tm1, a_t, r_t, s_t, game_over))
        
            if e > NUM_EPOCHS_OBSERVE:
                # finished observing, now start training
                # get next batch
                X, Y = get_next_batch(experience, model, NUM_ACTIONS, 
                                  GAMMA, BATCH_SIZE)
                loss += model.train_on_batch(X, Y)
        
        # reduce epsilon gradually
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS
        
        print("\rEpoch {:04d}/{:d} | Loss {:.5f} | Win Count: {:d}" 
              .format(e+1, NUM_EPOCHS, loss, num_wins), end="\r")
        fout.write("{:04d}\t{:.5f}\t{:d}\n" 
                   .format(e+1, loss, num_wins))

        if e % 100 == 0:
            model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
                
    print("")
    fout.close()
    model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
