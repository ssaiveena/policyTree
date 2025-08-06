import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from numba import njit
from policy import get_num_weights
from scipy.optimize import differential_evolution as DE
import pandas as pd
import json
import sys
sys.path.append('..')
from ptreeopt import PTreeOpt
import logging
from numba import njit
from oroville_example_reopt import train, simulate
import time
import random
from ptreeopt import PTreeOpt
import pickle
import numpy as gfg 
from ptreeopt import MPIExecutor
#from ptreeopt import MultiprocessingExecutor
# Example to run optimization and save results

pd.options.mode.chained_assignment = None  # default='warn'

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

@njit
def leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        year_leap=366
    else:
        year_leap=365
    return year_leap

cfs_to_tafd = 2.29568411 * 10 ** -5 * 86400 / 1000

def training_data(D, splitdata_demand, splitdata, window, years):
    '''to read the data for training based on the historical window number of years'''
    Q = pd.concat(splitdata[years + 50 - window : years + 50])
    T = len(Q)
    dowy = Q['dowy'].values
    demand = D[dowy] * pd.concat(splitdata_demand[years + 50 - window: years + 50])['combined_demand'].values

    return demand, Q, T, dowy

def wrapper_opt(P):

    cmip5_scenarios = pd.read_csv('cmip5/scenario_names_test.csv').name.to_list()
    lulc_scenarios = pd.read_csv('lulc/scenario_names.csv').name.to_list()
    storage_cost_tree = np.zeros(30*36)
    flood_cost_tree = np.zeros(30*36)
    
    ####################################################################################################
    ################################ Read indicator files ################################
    ####################################################################################################

    indicators = ['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50', 'dem_5', 'dem_10', 'storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10']

    data_ind = {}
    for indicator in indicators:
        data_ind[indicator] = pd.read_csv('indicator_files/%s_indicators.csv' % indicator)
            #indicators are saved from year 2000 similarlr df_Q and demand data are selected
    num = 0
    for gridi in range(30): #30#ranges upto 97
        for gridj in range(36): #ranges upto 36
            sc = cmip5_scenarios[gridi]
            sl = lulc_scenarios[gridj]
            df_Q = pd.read_csv('cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)
            df_demand = pd.read_csv('lulc/%s.csv.zip' % sl, index_col=0, parse_dates=True)
            df_Q['dowy'] = np.array([water_day_up(d, y) for d, y in zip(df_Q.index.dayofyear, df_Q.index.year)])

            splitdata = np.array_split(df_Q,
                                       np.intersect1d(np.where(df_Q.index.day == 1), np.where(df_Q.index.month == 10)))
            splitdata_demand = np.array_split(df_demand, np.intersect1d(np.where(df_demand.index.day == 1),
                                                                        np.where(df_demand.index.month == 10)))
            #finding number of years for simulation
            n_y = int(len(df_Q) / 365)
            storage_cost = np.zeros(n_y)
            flood_cost = np.zeros(n_y)

            storage_cost_perf = np.zeros(n_y)
            flood_cost_perf = np.zeros(n_y)

    
            '''reading historical trained policy to use in the beginning of timestep incase do nothing is activated
            in the first year. This trained policy is updated with time'''
            layers = np.array([3, 4, 1]) #defining the network structure with 3 inputs, 4 nodes and 1 layer
            num_weights = get_num_weights(layers)
            weight_bounds = [0, 1]

            res_rule = json.load(open('res_new.json'))
            
            ''' fixing few paramter values which remain constant in the future'''
            K = 3524  # capacity, TAF #this is fixed for the future too
            S_0 = K/2
            D = np.loadtxt('demand_Oroville.txt')  # target demand, TAF/d #demand is calculated based on median historical release multiplied by demand multiplier

            
            #historical demand remains the same in the future
            ####################################################################################################
            ################################ start looping through time series  ################################
            ####################################################################################################
            for years1 in range(50):
                storage_cost_perf[years1], flood_cost_perf[years1], S_0 = simulate(np.array(res_rule), layers, K, D[0:leap_year(1951+years1 + 1)] * splitdata_demand[years1+1]['combined_demand'].values, splitdata[years1+1]['ORO_inflow_cfs'].values * cfs_to_tafd,leap_year(1951+years1 + 1), splitdata[years1+1]['dowy'].values, S_0)

            data_ind['storage_cost_10'][sc][0] = np.mean(storage_cost_perf[years1-10: years1])
            data_ind['flood_cost_10'][sc][0] = np.mean(flood_cost_perf[years1-10: years1])
            
            data_ind['storage_cost_5'][sc][0] = np.mean(storage_cost_perf[years1 - 5: years1])
            data_ind['flood_cost_5'][sc][0] = np.mean(flood_cost_perf[years1 - 5: years1])
            
            S_0 = K/2
            
            for years in range(n_y-50):
                st = time.time()
                year_sim = 2000 + years
                policytree, rules = P.evaluate([data_ind['Peak_Flow_1'][sc][years],data_ind['Peak_Flow_5'][sc][years], data_ind['Peak_Flow_10'][sc][years], data_ind['MAF_5'][sc][years], data_ind['MAF_10'][sc][years],  data_ind['MAF_50'][sc][years], data_ind['dem_5'][sl][years], data_ind['dem_10'][sl][years], data_ind['storage_cost_5'][sc][years], data_ind['flood_cost_5'][sc][years], data_ind['storage_cost_10'][sc][years], data_ind['flood_cost_10'][sc][years]]) #indicators are saved from year 2000
                policytree = 'donothing'
                if policytree == 'donothing':
                    res_data = json.load(open('output/PF/res_%s_%s.json' %(sc, sl)))
                    res_rule = res_data[years] #res.x.tolist()
                    storage_cost[years], flood_cost[years], S_0 = simulate(np.array(res_rule), layers, K, D[0:leap_year(year_sim+1)] * splitdata_demand[years+50]['combined_demand'].values, splitdata[years+50]['ORO_inflow_cfs'].values * cfs_to_tafd, leap_year(year_sim+1), splitdata[years+50]['dowy'].values, S_0)
                elif policytree == 'Reopt_5' : #as an example reopt with 5year historical window
                    #Code to train policy for reoptimization of given historical window
                    window = 5
                    demand, Q, T, dowy = training_data(D, splitdata_demand, splitdata, window, years)
                    #res = DE(train, args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy), tol=1,  maxiter=10000,bounds=np.tile(weight_bounds, (num_weights,1)))
                    res_data = json.load(open('output/scenarios/w_%0.0f/res_%s_%s.json' %(5, sc, sl)))
                    '''update the policy for next year and to simulate in the current year'''
                    res_rule = res_data[years] #res.x.tolist()
                    storage_cost[years], flood_cost[years], S_0 = simulate(np.array(res_rule), layers, K, D[0:leap_year(year_sim+1)] * splitdata_demand[years+50]['combined_demand'].values, splitdata[years+50]['ORO_inflow_cfs'].values * cfs_to_tafd, leap_year(year_sim+1), splitdata[years+50]['dowy'].values, S_0)
                elif policytree == 'Reopt_10' : #as an example reopt with 5year historical window
                    #Code to train policy for reoptimization of given historical window
                    window = 10
                    demand, Q, T, dowy = training_data(D, splitdata_demand, splitdata, window, years)
                    #res = DE(train, args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy), tol=1,  maxiter=10000,bounds=np.tile(weight_bounds, (num_weights,1)))
                    res_data = json.load(open('output/scenarios/w_%0.0f/res_%s_%s.json' %(10, sc, sl)))
                    '''update the policy for next year and to simulate in the current year'''
                    res_rule = res_data[years] #res.x.tolist()
                    storage_cost[years], flood_cost[years], S_0 = simulate(np.array(res_rule), layers, K, D[0:leap_year(year_sim+1)] * splitdata_demand[years+50]['combined_demand'].values, splitdata[years+50]['ORO_inflow_cfs'].values * cfs_to_tafd, leap_year(year_sim+1), splitdata[years+50]['dowy'].values, S_0)
                elif policytree == 'Reopt_20' : #as an example reopt with 5year historical window
                    #Code to train policy for reoptimization of given historical window
                    window = 20
                    demand, Q, T, dowy = training_data(D, splitdata_demand, splitdata, window, years)
                    #res = DE(train, args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy), tol=1,  maxiter=10000,bounds=np.tile(weight_bounds, (num_weights,1)))
                    res_data = json.load(open('output/scenarios/w_%0.0f/res_%s_%s.json' %(20, sc, sl)))
                    '''update the policy for next year and to simulate in the current year'''
                    res_rule = res_data[years] #res.x.tolist()
                    storage_cost[years], flood_cost[years], S_0 = simulate(np.array(res_rule), layers, K, D[0:leap_year(year_sim+1)] * splitdata_demand[years+50]['combined_demand'].values, splitdata[years+50]['ORO_inflow_cfs'].values * cfs_to_tafd, leap_year(year_sim+1), splitdata[years+50]['dowy'].values, S_0)
                elif policytree == 'Reopt_50' : #as an example reopt with 5year historical window
                    #Code to train policy for reoptimization of given historical window
                    window = 50
                    demand, Q, T, dowy = training_data(D, splitdata_demand, splitdata, window, years)
                    #res = DE(train, args=(layers, K, demand, Q['ORO_inflow_cfs'].values * cfs_to_tafd, T, dowy), tol=1,  maxiter=10000,bounds=np.tile(weight_bounds, (num_weights,1)))
                    res_data = json.load(open('output/scenarios/w_%0.0f/res_%s_%s.json' %(50, sc, sl)))
                    '''update the policy for next year and to simulate in the current year'''
                    res_rule = res_data[years] #res.x.tolist()
                    storage_cost[years], flood_cost[years], S_0 = simulate(np.array(res_rule), layers, K, D[0:leap_year(year_sim+1)] * splitdata_demand[years+50]['combined_demand'].values, splitdata[years+50]['ORO_inflow_cfs'].values * cfs_to_tafd, leap_year(year_sim+1), splitdata[years+50]['dowy'].values, S_0)
                    
                    '''may save the policy if needed'''
                    # with open('res.json', 'w') as f:
                    #     json.dump(res.x.tolist(), f)

            storage_cost_tree[num] = storage_cost.sum()
            flood_cost_tree[num] = flood_cost.sum()
            num = num+1
            years1 = years1 + 1
            storage_cost_perf[years1] = storage_cost[years]
            flood_cost_perf[years1] = flood_cost[years]

            data_ind['storage_cost_10'][sc][years+1] = np.mean(storage_cost_perf[years1-10: years1])
            data_ind['flood_cost_10'][sc][years+1] = np.mean(flood_cost_perf[years1-10: years1])

            data_ind['storage_cost_5'][sc][years+1] = np.mean(storage_cost_perf[years1-5: years1])
            data_ind['flood_cost_5'][sc][years+1] = np.mean(flood_cost_perf[years1-5: years1])
    
#    data_ind['storage_cost_5'].to_csv('stor_5.txt')
#    data_ind['storage_cost_10'].to_csv('stor_10.txt')
#    data_ind['flood_cost_5'].to_csv('flood_5.txt')
#    data_ind['flood_cost_10'].to_csv('flood_10.txt')
#    
    #np.savetxt("flood_5.txt", data_ind['flood_cost_5'])
    #np.savetxt("storage_10.txt", data_ind['storage_cost_10'])
            
    #this returns the objectives of policy tree mean across scenarios
    return [storage_cost_tree.mean(), flood_cost_tree.mean()]

if __name__ == '__main__':
    for seed in range(1):
      snapshots = pickle.load(open('snapshots/result_perf_allscen_upind_%d.pkl' % seed, 'rb'), encoding='latin1')
      
      f = snapshots['best_f'][-1]
      P = snapshots['best_P'][-1]
      
      rows = len(P)
      arr = [[0 for i in range(2)] for j in range(rows)]
      
      for i in range(1):
        print(P[i])
        arr[i][0], arr[i][1] = wrapper_opt(P[i])
        print(arr[i][0])
        print(arr[i][1])
      #np.savetxt('result_perf_allscen_upind_%d.txt' %seed, arr)