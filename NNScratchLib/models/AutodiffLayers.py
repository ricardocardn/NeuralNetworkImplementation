import numpy as np
from scipy import signal

class SequentialLayer:
    def __init__(self, input, output, activation, activation_derivate):
        self.units = output
        self.weights = np.random.randn(output, input)
        self.bias = np.random.randn(output, 1)
        self.activation = activation
        self.activation_derivate = activation_derivate

    def z(self, A_prev):
        self.A_prev = A_prev
        return np.dot(self.weights, A_prev).reshape(self.units,1) + self.bias

    def A(self, A_prev):
        self.A_prev = A_prev
        self.output = self.activation(self.z(A_prev))
        return self.activation(self.z(A_prev))
    
    def backpropagation(self, dz, y=None, learning_rate=0.01):
        if dz.size == 0:
            self.output = self.output.reshape(len(self.output),1)
            y = y.reshape(len(y),1)

            dz = self.activation_derivate(self.output, y)
        else:
            dz = derivate = self.activation_derivate(self.z(self.A_prev))
            dz = derivate*dz
    
        dW = dz.dot(self.A_prev.T)
        db = dz
    
        dz_prev = self.weights.T.dot(dz)
    
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return dz_prev
    

class Convolutional():
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.weights[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.weights[i, j], "full")

        self.weights -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    

class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)