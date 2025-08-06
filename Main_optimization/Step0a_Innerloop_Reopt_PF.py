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

def training_data(D, splitdata_demand, splitdata, years):
    '''to read the data for training based on the historical window number of years'''
    Q = pd.concat(splitdata[years + 50 : years + 50+1])
    T = len(Q)
    dowy = Q['dowy'].values
    demand = D[dowy] * pd.concat(splitdata_demand[years + 50 : years + 50 + 1])['combined_demand'].values

    return demand, Q, T, dowy

cfs_to_tafd = 2.29568411 * 10 ** -5 * 86400 / 1000

def func(gridi,gridj):
    res_all = []
    cmip5_scenarios = pd.read_csv('cmip5/scenario_names.csv').name.to_list()
    lulc_scenarios = pd.read_csv('lulc/scenario_names.csv').name.to_list()
    filedir = 'output/PF/'
    sc = cmip5_scenarios[gridi]
    sl = lulc_scenarios[gridj]
    df_Q = pd.read_csv('cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)
    df_demand = pd.read_csv('lulc/%s.csv.zip' % sl, index_col=0, parse_dates=True)
    df_Q['dowy'] = np.array([water_day_up(d,y) for d,y in zip(df_Q.index.dayofyear,df_Q.index.year)])

    splitdata = np.array_split(df_Q, np.intersect1d(np.where(df_Q.index.day == 1), np.where(df_Q.index.month == 10)))
    splitdata_demand = np.array_split(df_demand, np.intersect1d(np.where(df_demand.index.day == 1), np.where(df_demand.index.month == 10)))
        # defining data for frequency "f"
    n_y = int(len(df_Q) / 365)

    layers = np.array([3, 4, 1]) #defining the network structure with 3 inputs, 4 nodes and 1 layer
    num_weights = get_num_weights(layers)
    weight_bounds = [0, 1]

    K = 3524  # capacity, TAF #this is fixed for the future too
    D = np.loadtxt('demand_Oroville.txt')  # target demand, TAF/d #demand is calculated based on median historical release multiplied by demand multiplier

    for years in range(n_y-50):
      year_sim = 2000 + years
      #Code to train policy for reoptimization of given historical window
      if years == 0:
        S_0 = K/2
      demand, Q, T, dowy = training_data(D, splitdata_demand, splitdata, years)

      #res = DE(train, args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy, S_0), tol= 1, maxiter=10000,bounds=np.tile(weight_bounds, (num_weights,1)))
      '''may save the policy if needed'''
      fun1 = []
      opt1 = {}
          
      for nopt in range(10):
        opt = DE(train, tol=1, maxiter=100000, bounds=np.tile(weight_bounds, (num_weights,1)), args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy, S_0))
        fun1.append(opt.fun)
        opt1[nopt] = opt.x.tolist()

      #res_all.append(res.x.tolist())
      res_all.append(opt1[fun1.index(min(fun1))])
    with open(filedir + 'res_%s_%s.json' %(sc, sl), 'w') as f:
      json.dump(res_all, f)

global S_0

st = time.time()
exp_a = range(30,97)
#range(6,7)#97) #(range(134, 154), 20) #134
exp_b = range(36)

# auxiliary funciton to make it work
def product_helper(args):
    return func(*args)

def parallel_product(list_a, list_b):
    #spark given number of processes
    p = Pool(128)
    # set each matching item into a tuple
    job_args = [(x, y) for x in list_a for y in list_b]
#[(item_a, list_b[i]) for i, item_a in enumerate(list_a)]
    # map to pool
    p.map(product_helper, job_args)

if __name__ == '__main__':
    parallel_product(exp_a, exp_b)

    et  = time.time()
    print(et-st)
