import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.optimize import differential_evolution as DE
from policy import get_num_weights
import time
from numba import njit
import math
import sys
sys.path.append('..')
from multiprocessing import Pool
import os
from innerloop_train import train, simulate

@njit
def water_day_up(d,year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        year_leap=1
    else:
        year_leap=0
    if d>=(274+year_leap):
        yearday = d - (274+year_leap)
    else:
        yearday = d + 91
    return yearday

def training_data(D, splitdata_demand, splitdata, window, years):
    '''to read the data for training based on the historical window number of years'''
    Q = pd.concat(splitdata[years + 50 - window : years + 50])
    T = len(Q)
    dowy = Q['dowy'].values
    demand = D[dowy] * pd.concat(splitdata_demand[years + 50 - window: years + 50])['combined_demand'].values

    return demand, Q, T, dowy

cfs_to_tafd = 2.29568411 * 10 ** -5 * 86400 / 1000

def func(gridi,gridj,w):
    print('a')
    res_all = []
    cmip5_scenarios = pd.read_csv('../Data/cmip5/scenario_names.csv').name.to_list()
    lulc_scenarios = pd.read_csv('../Data/lulc/scenario_names.csv').name.to_list()
    filedir = 'output/scenarios/' + 'w_' + str(w) + '/'
    sc = cmip5_scenarios[gridi]
    sl = lulc_scenarios[gridj]
    df_Q = pd.read_csv('../Data/cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)
    df_demand = pd.read_csv('../Data/lulc/%s.csv.zip' % sl, index_col=0, parse_dates=True)
    df_Q['dowy'] = np.array([water_day_up(d,y) for d,y in zip(df_Q.index.dayofyear,df_Q.index.year)])
    print('a')
    splitdata = np.array_split(df_Q, np.intersect1d(np.where(df_Q.index.day == 1), np.where(df_Q.index.month == 10)))
    splitdata_demand = np.array_split(df_demand, np.intersect1d(np.where(df_demand.index.day == 1), np.where(df_demand.index.month == 10)))
        # defining data for frequency "f"
    n_y = int(len(df_Q) / 365)

    layers = np.array([3, 4, 1]) #defining the network structure with 3 inputs, 4 nodes and 1 layer
    num_weights = get_num_weights(layers)
    weight_bounds = [0, 1]

    K = 3524  # capacity, TAF #this is fixed for the future too
    D = np.loadtxt('demand_Oroville.txt')  # target demand, TAF/d #demand is calculated based on median historical release multiplied by demand multiplier
    print('a')
    for years in range(n_y-50):
      year_sim = 2000 + years
      #Code to train policy for reoptimization of given historical window
      window = w
      print('a')
      demand, Q, T, dowy = training_data(D, splitdata_demand, splitdata, window, years)

      res = DE(train, args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy), tol= 1, maxiter=10000,bounds=np.tile(weight_bounds, (num_weights,1)))

      '''may save the policy if needed'''
      print('a')
      res_all.append(res.x.tolist())
    with open(filedir + 'res_%s_%s.json' %(sc, sl), 'w') as f:
      print('a')
      json.dump(res_all, f)


w_comb = [50]

st = time.time()
exp_a = range(1)#97
exp_b = range(1)#36

# auxiliary funciton to make it work
# def product_helper(args):
#     return func(*args)
#
# def parallel_product(list_a, list_b,w):
#     #spark given number of processes
#     p = Pool(128)
#     # set each matching item into a tuple
#     job_args = [(x, y, w) for x in list_a for y in list_b]
# #[(item_a, list_b[i]) for i, item_a in enumerate(list_a)]
#     # map to pool
#     p.map(product_helper, job_args)

if __name__ == '__main__':
    # for w in w_comb:
    #     parallel_product(exp_a, exp_b, w)

    func(0,0,50)
    et  = time.time()
    print(et-st)
