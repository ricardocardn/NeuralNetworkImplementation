import numpy as np

def sigmoid(x):
  return 1/(1+np.e**(-x))

def sigmoid_derivate(x):
  return sigmoid(x)*(1 - sigmoid(x))

def ReLU(x):
  return np.maximum(0, x)

def ReLU_derivate(x):
  return x > 0

def softmax(A):
  exp_A = np.e**(A - np.max(A))
  return exp_A / np.sum(exp_A)

def softmax_derivate(Y_pred, Y):
  return Y_pred - Y

# error for autoencoder
def mse(Y_pred, Y):
  return np.mean((Y_pred - Y)**2)

def mse_derivate(Y_pred, Y):
  return (Y_pred - Y)*sigmoid_derivate(Y_pred)