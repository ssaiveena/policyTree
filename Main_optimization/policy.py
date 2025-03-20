import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from numba import njit

@njit
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

@njit
def relu(x):
  a = np.zeros(x.shape, dtype=np.float32)
  return np.maximum(a,x)

@njit
def get_num_weights(layers):
  nw = 0
  for i in range(len(layers)-1):
    nw += layers[i] * layers[i+1]
  return nw 


# evaluate policy for a set of inputs and weights
# todo: allow different activations
@njit
def policy(inputs, weights, layers):
  
  start = 0

  for i in range(len(layers)-1): 
    # extract and reshape weights for each layer 
    w = weights[start:start+layers[i]*layers[i+1]].reshape(layers[i],layers[i+1])
    start += layers[i]*layers[i+1]

    if i == 0: # first layer
      z = relu(inputs @ w)
    else:
      z = relu(z @ w)

  return z[0]