import pickle
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = list()

    def set(self, layer):
        self.layers.append(layer)

    def feedfoward(self, X, n=None):
        n= len(self.layers) if not n else n

        for layer in self.layers[:n]:
            X = layer.A(X)

        return X
    
    def backward(self, X, Y, acc_measure, learning_rate, epochs=1000):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                x = x.reshape(len(x),1)
                y = y.reshape(len(y),1)

                result = self.feedfoward(x)

                dz = self.layers[-1].backward(np.array([]), y)

                for layer in self.layers[-2::-1]:
                    dz = layer.backward(dz, learning_rate=learning_rate)

            if epoch % 10 == 0:
                Y_pred = [self.feedfoward(x) for x in X]
                print(f"Epoch: {epoch:4.0f}, Accuracy: {acc_measure(Y_pred, Y):.3f}")
    
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load_data(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)