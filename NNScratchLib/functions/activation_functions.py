import numpy as np

def sigmoid(x):
  return 1/(1+np.e**(-x))

def sigmoid_derivate(x):
  return sigmoid(x)*(1 - sigmoid(x))

def ReLU(x):
  return np.maximum(0, x)

def ReLU_derivate(x):
  return x > 0

def leaky_ReLU(x):
  return np.maximum(0.01*x, x)

def leaky_ReLU_derivate(x):
  return 0.01 if x < 0 else 1

def tanh(x):
  return np.tanh(x)

def tanh_derivate(x):
  return 1 - np.tanh(x)**2

def identity(x):
  return x

def identity_derivate(x):
  return 1

def softmax(A):
  exp_A = np.e**(A - np.max(A))
  return exp_A / np.sum(exp_A)

def softmax_derivate(Y_pred, Y):
  return Y_pred - Y

def mse(Y_pred, Y):
  return np.mean((Y - Y_pred)**2)

def mse_derivate(Y_pred, Y):
  return -2 * (Y - Y_pred)*Y_pred*(1 - Y_pred)

def mse_derivate2(Y_pred, Y):
  return 2 * (Y_pred - Y)