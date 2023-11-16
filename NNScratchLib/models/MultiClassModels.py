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
        A = self.feedfoward(X)
        A = A.reshape(len(A),1)
        y = y.reshape(len(y),1)

        # Calculamos el error de la última capa
        A_prev = self.feedfoward(X, -1)
        A_prev = A_prev.reshape(len(A_prev),1)

        dz = A - y
        dW = dz.dot(A_prev.T)

        # Calculamos el error de la capa anterior
        A_prev_2 = self.feedfoward(X, -2)
        A_prev_2 = A_prev_2.reshape(len(A_prev_2),1)

        derivate = self.layers[-2].activation_derivate(self.layers[-2].z(A_prev_2))
        dz2 = derivate*self.layers[-1].weights.T.dot(dz)
        dW2 = dz2.dot(A_prev_2.T)

        return dz, dW, dz2, dW2