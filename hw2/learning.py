from sklearn.datasets import make_moons
from sklearn.cross_validation import train_test_split
import numpy as np
from random import shuffle

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
n_feature = 2
n_class = 2
n_iter = 10

alpha = 0.01

def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def make_network(n_hidden=100):
    # Initialize weights with Standard Normal random variables

    # It depends on how many layers we have. W1 connects the n_feature to n_hidden, so 2 to 100.
    # But W2 is from hidden to classifier, so we should have 100 to 2.
    model = dict(
        # it seems like W1 is 2 * 100
        W1=np.random.randn(n_feature, n_hidden),
        # and W2 is 100 * 2
        W2=np.random.randn(n_hidden, n_class)
    )

    return model

def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    # ReLU non-linearity
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h @ model['W2'])

    return h, prob

def backward(model, xs, hs, errs):
    """xs, hs, errs contain all informations (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    dW2 = hs.T @ errs

    # Get gradient of hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0

    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)

def momentum(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    print(velocity)
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model

def get_minibatch(X, y, minibatch_size):
    minibatches = []

    # X, y = shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches

def sgd(model, X_train, y_train, minibatch_size):
    print(model)
    for iter in range(n_iter):
        print('Iteration {}'.format(iter))

        # Randomize data point
        # X_train, y_train = shuffle(X_train, y_train)

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

            model = sgd_step(model, X_train_mini, y_train_mini)

    return model

def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Learning rate: 1e-4
        model[layer] += 1e-4 * grad[layer]

    return model

def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        # Create probability distribution of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.

        # Compute the gradient of output layer
        err = y_true - y_pred

        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))


minibatch_size = 50
n_experiment = 100

# Create placeholder to accumulate prediction accuracy
accs = np.zeros(n_experiment)

for k in range(n_experiment):
    # Reset model
    model = make_network()

    # Train the model
    model = momentum(model, X_train, y_train, minibatch_size)

    y_pred = np.zeros_like(y_test)

    for i, x in enumerate(X_test):
        # Predict the distribution of label
        _, prob = forward(x, model)
        # Get label by picking the most probable one
        y = np.argmax(prob)
        y_pred[i] = y

    # Compare the predictions with the true labels and take the percentage
    accs[k] = (y_pred == y_test).sum() / y_test.size

print('Mean accuracy: {}, std: {}'.format(accs.mean(), accs.std()))
