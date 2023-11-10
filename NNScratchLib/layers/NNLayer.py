import numpy as np

class Layer:
    def __init__(self, input, output, activation, activation_derivate):
        self.units = output
        self.weights = np.random.randn(output, input)
        self.bias = np.random.randn(output, 1)
        self.activation = activation
        self.activation_derivate = activation_derivate

    def z(self, A_prev):
        return np.dot(self.weights, A_prev).reshape(self.units,1) + self.bias

    def A(self, A_prev):
        return self.activation(self.z(A_prev))