import numpy as np
import pandas as pd
from numba import njit
from policy import *

@njit
def train(weights, layers, K, D, Q, T, dowy, S_0):
    short, flood, S_0 = simulate(weights, layers, K, D, Q, T, dowy, S_0)
    c = short + flood
    return c


@njit
def simulate(weights, layers, K, D, Q, T, dowy, S_0):
    S = np.zeros(T)
    R = np.zeros(T)
    storage_cost = np.zeros(T)
    flood_cost = np.zeros(T)
    inputs_save = np.zeros((T, 3))
    target = np.zeros(T)
    Q_scaling_factor = 100  # TAF/d

    #S[0] = K / 2
    inputs = np.zeros(layers[0])

    for t in range(0, T): #(1, T)
        if t != 0:
          S[t] = S[t - 1] + Q[t - 1] - R[t - 1]
        else:
          S[t] = S_0 + Q[t - 1] - R[t - 1]
        W = S[t] + Q[t]

        inputs[0] = S[t] / K  # storage fraction of capacity

        if inputs.size >= 2:
            inputs[1] = -np.cos(2 * np.pi * dowy[t] / 365)  # transform dowy to work with relu
        if inputs.size >= 3:
            inputs[2] = Q[t:t + 5].mean() / Q_scaling_factor  # average 5-day forecast, scaled
        if inputs.size >= 4:
            inputs[3] = Q[t:t + 90].mean() / Q_scaling_factor  # average seasonal forecast, scaled
        # this last input is unrealistic, we don't have seasonal predictability
        inputs_save[t,:] = inputs
        P = policy(inputs, weights, layers)

        # policy scaling - piecewise linear
        # think of this as a custom activation function
        # because the flood releases need to be so much larger
        # it works better than linearly scaling the policy output
        if P < 0.5:  # hedging
            target[t] = (0.5 + P) * D[t]
        else:  # flood control
            # target[t] = P * 100
            target[t] = 2 * (Q_scaling_factor - D[t]) * P + (-1 * Q_scaling_factor + 2 * D[t])

        R[t] = max(target[t], 0)  # no negative release

        if R[t] > W:  # release limited by water available
            R[t] = W
        if S[t] > K:  # spill
            R[t] += S[t] - K

        # objective - drought + flood costs
        # flood cost is a large penalty for > 100 tafd release
        # but not so large that it drowns out the shortage signal
        storage_cost[t] = max(D[t] - R[t], 0) ** 2
        flood_cost[t] = 10 ** 2 * max(R[t] - 100, 0)

    return storage_cost.sum(), flood_cost.sum(), S[t] , S, R, inputs_save
