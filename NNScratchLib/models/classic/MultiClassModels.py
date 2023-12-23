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
    
    def backpropagation(self, X, y):
        derivates = []

        A = self.feedfoward(X)
        A = A.reshape(len(A),1)
        y = y.reshape(len(y),1)

        A_prev = self.feedfoward(X, -1)
        A_prev = A_prev.reshape(len(A_prev),1)

        dz = self.layers[-1].activation_derivate(A, y)
        dW = dz.dot(A_prev.T)

        derivates.append((dz, dW))

        for k, layer in enumerate(self.layers[-2::-1]):
            A_prev = self.feedfoward(X, -k-2)
            A_prev = A_prev.reshape(len(A_prev),1)

            derivate = layer.activation_derivate(layer.z(A_prev))
            dz = derivate*self.layers[-k-1].weights.T.dot(dz)
            dW = dz.dot(A_prev.T)

            derivates.append((dz, dW))

        return derivates
    
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load_data(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)