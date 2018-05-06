# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import os

import paddle_game

from rl_game_train import preprocess_frames

############################# main ###############################

DATA_DIR = ""

BATCH_SIZE = 32
NUM_EPOCHS = 100

modelFile = "rl-network.h6"
model = load_model(os.path.join(DATA_DIR, modelFile))
model.compile(optimizer=Adam(lr=1e-6), loss="mse")

# train network
game = paddle_game.PaddleGame()

num_games, num_wins = 0, 0
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
        q = model.predict(s_t)[0]
        a_t = np.argmax(q)
        # apply action, get reward
        x_t, r_t, game_over = game.step(a_t)
        s_t = preprocess_frames(x_t)
        # if reward, increment num_wins
        if r_t == 1: num_wins += 1

    num_games += 1
    print("\rGame: {:03d}, Wins: {:03d}"
          .format(num_games, num_wins), end="\r")
        
print("")
