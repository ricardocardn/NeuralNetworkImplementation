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