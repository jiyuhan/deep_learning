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

"""
This function requests a batch and returns the batch
"""
def get_next_batch(experience, model, num_actions, gamma, batch_size):
    batch_indices = np.random.randint(low=0, high=len(experience),
                                      size=batch_size)
    batch = [experience[i] for i in batch_indices]
    # batch = experience[batch_indices]
    # batch is a list of experiences: s_t, a_t, r_t, s_tp1, game_over
    X = np.zeros((batch_size, 3, 4, 1))
    # X is a batch_size of frames.
    Y = np.zeros((batch_size, num_actions))
    # Y is a batch size of rewards (for each action).
    for i in range(len(batch)):
        x_tm1, x_t, r_0, game_over, won = batch[i]
        X[i] = x_t
        Y[i] = model.predict(x_tm1)[0]
        Q_sa = np.max(model.predict(x_t)[0])
        if game_over:
            Y[i, a_t] = r_t
        else:
            Y[i, a_t] = r_t + gamma * Q_sa
    return X, Y

###################################


if __name__ == "__main__":
    # initialize parameters
    DATA_DIR = ""
    NUM_ACTIONS = 3  # 0 (stand); 1 (hit); 2 (double)
    GAMMA = 0.99  # decay rate of past observations
    INITIAL_EPSILON = 0.1  # starting value of epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    MEMORY_SIZE = 50000  # number of previous transitions to remember
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

        # x_t: current_state[dealer_hand, player_hand],
        # r_0: reward is funds,
        # game_over: boolean,
        # won: 1 for win, -1 for loss, 0 for tie
        x_t, r_0, game_over, won = game.oneStep(a_0)

        while not game_over:
            x_tm1 = x_t
            # next action
            if e <= NUM_EPOCHS_OBSERVE or np.random.rand() <= epsilon:
                a_t = np.random.randint(low=0, high=NUM_ACTIONS)
            else:
                q = model.predict(x_t)
                a_t = np.argmax(q)

            # apply action, get reward
            x_t, r_t, game_over, won = game.oneStep(a_t)

            # if game won, increment num_wins
            if won == 1:
                num_wins += 1
            # store experience
            experience.append((x_tm1, x_t, r_0, game_over, won))

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
              .format(e + 1, NUM_EPOCHS, loss, num_wins), end="\r")
        fout.write("{:04d}\t{:.5f}\t{:d}\n"
                   .format(e + 1, loss, num_wins))

        if e % 100 == 0:
            model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
    print("")
    fout.close()
    model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)