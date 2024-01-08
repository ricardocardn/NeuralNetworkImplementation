import numpy as np
from matplotlib import pyplot as plt

def mean_squared_error(Y_pred, Y):
    sse = (Y - Y_pred)**2
    mse = np.mean(sse)
    return mse 


def print_images(init_pos, n, x_train, model, save=False, name=None):
    plt.figure(figsize=(19, 4))
    plt.gray()

    output = [model.feedfoward(x_train[i]).reshape(28, 28) for i in range(init_pos, init_pos + n)]
    
    for i, item in enumerate(x_train[init_pos:init_pos + n]):
        plt.subplot(2, n, i + 1)
        plt.title(f'Original')
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    for i, item in enumerate(output):
        plt.subplot(2, n, n + i + 1)
        plt.title(f'Generated')
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    if save == True and name != None: plt.savefig(f'images/{name}.png')
    else: plt.show()
    plt.close()


def gradient_descent_autoenc(model, X, y, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X)):
            derivates = model.backpropagation(X[i], y[i])                
            for layer, derivate in zip(model.layers[::-1], derivates):
                layer.weights -= learning_rate*derivate[1]
                layer.bias -= learning_rate*derivate[0]

        if epoch % 10 == 0:
            print_images(epoch, 10, X_val, model, save=True, name=f'denoising/training_epoch_{epoch}')

            loss = mean_squared_error(model.feedfoward(X[i]), y[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}')

            loss_list.append(loss)

    return loss_list


def gradient_descent_regression(model, X, y, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    loss_list = []

    loss = mean_squared_error(model.feedfoward(X[0]), y[0])
    print(f'Loss before training {loss:.5f}')
    loss_list.append(loss)

    for epoch in range(epochs):
        for i in range(len(X)):
            derivates = model.backpropagation(X[i], y[i])                
            for layer, derivate in zip(model.layers[::-1], derivates):
                layer.weights -= learning_rate*derivate[1]
                layer.bias -= learning_rate*derivate[0]

        if epoch % 10 == 0:
            loss = mean_squared_error(model.feedfoward(X[i]), y[i])
            print(f'epoch {epoch:3} - Loss {loss:.5f}')
            loss_list.append(loss)

    return loss_list