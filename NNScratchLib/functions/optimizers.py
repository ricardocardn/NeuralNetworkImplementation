import numpy as np
from matplotlib import pyplot as plt

def cross_entropy(Y_pred, Y):
    return -np.sum(Y*np.log(Y_pred.reshape(1, len(Y_pred))))


def gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X)):
            derivates = model.backpropagation(X[i], y[i])                
            for layer, derivate in zip(model.layers[::-1], derivates):
                layer.weights -= learning_rate*derivate[1]
                layer.bias -= learning_rate*derivate[0]

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)

            loss = cross_entropy(model.feedfoward(X[i]), y[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list, loss_list


def momentum_gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    vW = [np.zeros(layer.weights.shape) for layer in model.layers]
    vb = [np.zeros(layer.bias.shape) for layer in model.layers]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X)):
            derivates = model.backpropagation(X[i], y[i]) 

            pos = len(model.layers)-1           
            for layer, derivate in zip(model.layers[::-1], derivates):
                vW[pos] = 0.9*vW[pos] + learning_rate*derivate[1]
                vb[pos] = 0.9*vb[pos] + learning_rate*derivate[0]

                layer.weights -= vW[pos]
                layer.bias -= vb[pos]

                pos -= 1

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)

            loss = cross_entropy(model.feedfoward(X[i]), y[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list


def Adam(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    vW = [np.zeros(layer.weights.shape) for layer in model.layers]
    vb = [np.zeros(layer.bias.shape) for layer in model.layers]

    sW = [np.zeros(layer.weights.shape) for layer in model.layers]
    sb = [np.zeros(layer.bias.shape) for layer in model.layers]

    acc_list = []
    loss_list = []
    
    for epoch in range(epochs):
        for i in range(len(X)):
            derivates = model.backpropagation(X[i], y[i]) 

            pos = len(model.layers)-1           
            for layer, derivate in zip(model.layers[::-1], derivates):
                vW[pos] = 0.9*vW[pos] + (1-0.9)*derivate[1]
                vb[pos] = 0.9*vb[pos] + (1-0.9)*derivate[0]

                sW[pos] = 0.999*sW[pos] + (1-0.999)*derivate[1]**2
                sb[pos] = 0.999*sb[pos] + (1-0.999)*derivate[0]**2

                vW_corrected = vW[pos]/(1-0.9**(epoch+1))
                vb_corrected = vb[pos]/(1-0.9**(epoch+1))

                sW_corrected = sW[pos]/(1-0.999**(epoch+1))
                sb_corrected = sb[pos]/(1-0.999**(epoch+1))

                layer.weights -= learning_rate*vW_corrected/(np.sqrt(sW_corrected) + 1e-8)
                layer.bias -= learning_rate*vb_corrected/(np.sqrt(sb_corrected) + 1e-8)

                pos -= 1

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)

            loss = cross_entropy(model.feedfoward(X[i]), y[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)