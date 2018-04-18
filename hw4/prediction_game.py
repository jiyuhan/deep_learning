import sys
import matplotlib.pyplot as plt
from datasets import generate_x_y_data_v5
import numpy as np
import tensorflow as tf

generate_x_y_data = generate_x_y_data_v5


def play_the_game():
    cash = 100000
    shares = 0
    game_data = get_300_day_data()


def should_buy(x0, x1, x2, x3, x4, x5):
    if x0 < (x1+x2)/2 < (x3+x4)/2 < x5:
        return True
    else:
        return False


def buy_or_sell(x0, x1, x2, x3, x4, x5):
    if x0 > (x1+x2)/2 > (x3+x4)/2 > x5:
        return 0 # sell
    elif x0 < (x1+x2)/2 < (x3+x4)/2 < x5:
        return 1 # buy
    return 2


def buy_order(cash, shares, price):
    if cash < 10000:
        return cash, shares  # buy
    buy_share = 10000 // price
    return cash - (buy_share * price), shares + buy_share


def sell_order(cash, shares, price):
    sell_share = shares // 3
    return cash + sell_share * price, shares - sell_share


def get_300_day_data():
    # something something
    x, _ = generate_x_y_data(
        isTest=0, batch_size=batch_size, predict_days=predict_days, load_purpose=1)
    return x


def predict_five_day(x0_actual, x1_actual, x2_actual, x3_actual, x4_actual, x5_actual):

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

    # j is how many predictions we are making
    # in our case, we are predicting five days
    # in its original use, every prediction plots a figure.
    for j in range(num_predictions):
        plt.figure(figsize=(12, 3))

        for k in range(output_dim):
            past = X[:, j, k]
            past = past[:predict_days]
            expected = Y[:, j, k]
            pred = outputs[:, j, k]

            label1 = "Seen (past) values" if k == 0 else "_nolegend_"
            label2 = "True future values" if k == 0 else "_nolegend_"
            label3 = "Predictions" if k == 0 else "_nolegend_"
            plt.plot(range(len(past)), past, "o--b", label=label1)
            plt.plot(range(len(past), len(expected)+len(past)),
                     expected, "x--b", label=label2)
            plt.plot(range(len(past), len(pred)+len(past)),
                     pred, "o--y", label=label3)

        plt.legend(loc='best')
        plt.title("Predictions v.s. true values")
        plt.show()



def train_data(x_train, y_train):

    x = loadStockData()
    return 1


def loadStockData():
    return 1

# Training of the neural net


def train_batch(batch_size):
    """
    Training step that optimizes the weights 
    provided some batch_size X and Y examples from the dataset. 
    """
    X, Y = generate_x_y_data(
        isTest=0, batch_size=batch_size, predict_days=predict_days)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t]
                      for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op. 
    """
    X, Y = generate_x_y_data(
        isTest=1, batch_size=batch_size, predict_days=predict_days)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t]
                      for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


if __name__ == "__main__":

    # This is for the notebook to generate inline matplotlib
    # charts rather than to open a new window every time:
    # get_ipython().magic('matplotlib inline')

    # ## Neural network's hyperparameters

    # In[3]:
    # Configuration of Prediction:
    num_predictions = 4
    predict_days = 5  # prediction in the next predict_days

    # Configuration of Optmizer:
    learning_rate = 0.001  # Small lr helps not to diverge during training.
    # How many times we perform a training step (therefore how many times we show a batch).
    num_iters = 500
    lr_decay = 0.92  # default: 0.9 . Simulated annealing.
    momentum = 0.5  # default: 0.0 . Momentum technique in weights update
    lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting
    batch_size = 100  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

    # Neural network parameters
    hidden_dim = 200  # Count of hidden neurons in the recurrent units.
    # Number of stacked recurrent cells, on the neural depth axis.
    layers_stacked_count = 2

    sample_x, sample_y = generate_x_y_data(
        isTest=0, batch_size=batch_size, predict_days=predict_days)
    print("Dimensions of X and Y training examples: ")
    print("  (seq_length, batch_size, output_dim) = ",
          sample_x.shape, sample_y.shape)

    # Dependent neural network parameters
    seq_length = sample_x.shape[0]  # Time series for backpropagation
    # Output dimension (e.g.: multiple signals at once, tied in time)
    output_dim = input_dim = sample_x.shape[-1]

    # ## Definition of the seq2seq neuronal architecture
    #
    # <img src="https://www.tensorflow.org/images/basic_seq2seq.png" />
    #
    # Comparatively to what we see in the image, our neural network deals with signal rather than letters. Also, we don't have the feedback mechanism yet.

    # Backward compatibility for TensorFlow's version 0.12:
    try:
        tf.nn.seq2seq = tf.contrib.legacy_seq2seq
        tf.nn.rnn_cell = tf.contrib.rnn
        tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
        # print("TensorFlow's version : 1.0 (or more)")
    except:
        print("TensorFlow's version : 0.12")

    tf.reset_default_graph()
    # sess.close()
    sess = tf.InteractiveSession()

    with tf.variable_scope('Seq2seq'):

        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(
                None, input_dim), name="inp_{}".format(t))
            for t in range(seq_length)
        ]

        # Decoder: expected outputs
        expected_sparse_output = [
            tf.placeholder(tf.float32, shape=(None, output_dim),
                           name="expected_sparse_output_{}".format(t))
            for t in range(seq_length)
        ]

        # Give a "GO" token to the decoder.
        # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ enc_inp[:-1]"
        dec_inp = [tf.zeros_like(
            enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]
        # dec_inp = enc_inp

        # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
        cells = []
        for i in range(layers_stacked_count):
            with tf.variable_scope('RNN_{}'.format(i)):
                cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
                # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # Here, the encoder and the decoder uses the same cell, HOWEVER,
        # the weights aren't shared among the encoder and decoder, we have two
        # sets of weights created under the hood according to that function's def.
        dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell
        )

        # For reshaping the output dimensions of the seq2seq RNN:
        w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
        b_out = tf.Variable(tf.random_normal([output_dim]))

        # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
        output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")

        reshaped_outputs = [output_scale_factor *
                            (tf.matmul(i, w_out) + b_out) for i in dec_outputs]

    # Training loss and optimizer

    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
            output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

        # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=lr_decay, momentum=momentum)
        train_op = optimizer.minimize(loss)

    # Training
    train_losses = []
    test_losses = []

    sess.run(tf.global_variables_initializer())
    for t in range(num_iters+1):
        train_loss = train_batch(batch_size)
        train_losses.append(train_loss)

        if t % 10 == 0:
            # Tester
            test_loss = test_batch(batch_size)
            test_losses.append(test_loss)
            sys.stdout.flush()
            sys.stdout.write("\rStep %d/%d, train loss: %.2f, \tTEST loss: %.2f" %
                             (t, num_iters, train_loss, test_loss))
            #print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, num_iters, train_loss, test_loss))

    print("\nFinal train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
    print('training finished, start to play the game')


    play_the_game()
