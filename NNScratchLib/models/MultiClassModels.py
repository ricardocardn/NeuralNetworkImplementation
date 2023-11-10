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
    
    def backpropagation(self, X, y, lr=0.01):
        A = self.feedfoward(X)
        A = A.reshape(len(A),1)
        y = y.reshape(len(y),1)

        dz = A - y

        A_prev = self.feedfoward(X, -1)
        A_prev = A_prev.reshape(len(A_prev),1)

        dW = dz.dot(A_prev.T)
        self.layers[-1].weights -= dW
        self.layers[-1].bias -= dz

        for k, layer in enumerate(self.layers[-2::-1]):
            A_prev = self.feedfoward(X, -k-2)
            A_prev = A_prev.reshape(len(A_prev),1)

            derivate = layer.activation_derivate(layer.z(A_prev))
            dz = derivate*self.layers[-k-1].weights.T.dot(dz)
            layer.bias -= lr*dz
            
            dW = dz.dot(A_prev.T)
            layer.weights -= lr*dW