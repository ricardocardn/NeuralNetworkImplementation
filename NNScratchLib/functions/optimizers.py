import numpy as np

def gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    for epoch in range(epochs):
        for i in range(len(X)):
            dz, dW, dz2, dW2 = model.backpropagation(X[i], y[i])                
            model.layers[-1].weights -= learning_rate*dW
            model.layers[-2].weights -= learning_rate*dW2

            model.layers[-1].bias -= learning_rate*dz
            model.layers[-2].bias -= learning_rate*dz2

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)
            print(f'epoch {epoch:3} - accuracy {acc:.5f}')


def momentum_gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    vW = [np.zeros(layer.weights.shape) for layer in model.layers]
    vb = [np.zeros(layer.bias.shape) for layer in model.layers]

    for epoch in range(epochs):
        for i in range(len(X)):
            dz, dW, dz2, dW2 = model.backpropagation(X[i], y[i])                
            vW[-1] = 0.9*vW[-1] + learning_rate*dW
            vW[-2] = 0.9*vW[-2] + learning_rate*dW2

            vb[-1] = 0.9*vb[-1] + learning_rate*dz
            vb[-2] = 0.9*vb[-2] + learning_rate*dz2

            model.layers[-1].weights -= vW[-1]
            model.layers[-2].weights -= vW[-2]

            model.layers[-1].bias -= vb[-1]
            model.layers[-2].bias -= vb[-2]

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)
            print(f'epoch {epoch:3} - accuracy {acc:.5f}')