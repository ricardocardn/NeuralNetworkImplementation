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

        # Calculamos el error de la última capa
        error = A - y

        A_prev = self.feedfoward(X, -1)
        A_prev = A_prev.reshape(len(A_prev),1)

        # actualizamos los pesos de la última capa
        dW = error.dot(A_prev.T)

        self.layers[-1].weights -= dW
        self.layers[-1].bias -= error

        # Calculamos el error de la capa anterior
        A_prev_2 = self.feedfoward(X, -2)
        A_prev_2 = A_prev_2.reshape(len(A_prev_2),1)

        derivate = self.layers[-2].activation_derivate(self.layers[-2].z(A_prev_2))
        dz2 = derivate*self.layers[-1].weights.T.dot(error)
        self.layers[-2].bias -= lr*dz2
        
        dz2 = dz2.dot(A_prev_2.T)
        self.layers[-2].weights -= lr*dz2