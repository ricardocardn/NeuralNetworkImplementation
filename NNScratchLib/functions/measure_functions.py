import numpy as np

def accuracy(Y_pred, Y):
    acc = 0
    for y_pred, y in zip(Y_pred, Y):
        if np.argmax(y_pred) == np.argmax(y):
            acc += 1

    return acc / len(Y_pred)