import numpy as np

def gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    acc_list = []
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
            acc_list.append(acc)

    return acc_list


def momentum_gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    vW = [np.zeros(layer.weights.shape) for layer in model.layers]
    vb = [np.zeros(layer.bias.shape) for layer in model.layers]

    acc_list = []
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
            acc_list.append(acc)

    return acc_list


def Adam(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    vW = [np.zeros(layer.weights.shape) for layer in model.layers]
    vb = [np.zeros(layer.bias.shape) for layer in model.layers]

    sW = [np.zeros(layer.weights.shape) for layer in model.layers]
    sb = [np.zeros(layer.bias.shape) for layer in model.layers]

    acc_list = []
    for epoch in range(epochs):
        for i in range(len(X)):
            dz, dW, dz2, dW2 = model.backpropagation(X[i], y[i])
            params = [dW2, dz2, dW, dz]

            for layer in range(len(model.layers)):
                vW[layer] = 0.9*vW[layer] + (1-0.9)*params[2*layer]
                vb[layer] = 0.9*vb[layer] + (1-0.9)*params[2*layer+1]

                sW[layer] = 0.999*sW[layer] + (1-0.999)*params[2*layer]**2
                sb[layer] = 0.999*sb[layer] + (1-0.999)*params[2*layer+1]**2

                vW_corrected = vW[layer]/(1-0.9**(epoch+1))
                vb_corrected = vb[layer]/(1-0.9**(epoch+1))

                sW_corrected = sW[layer]/(1-0.999**(epoch+1))
                sb_corrected = sb[layer]/(1-0.999**(epoch+1))

                model.layers[layer].weights -= learning_rate*vW_corrected/(np.sqrt(sW_corrected) + 1e-8)
                model.layers[layer].bias -= learning_rate*vb_corrected/(np.sqrt(sb_corrected) + 1e-8)

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)
            print(f'epoch {epoch:3} - accuracy {acc:.5f}')
            acc_list.append(acc)

    return acc_list