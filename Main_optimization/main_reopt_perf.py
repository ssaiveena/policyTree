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
#from ptreeopt import MPIExecutor
from ptreeopt import MultiprocessingExecutor
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

    cmip5_scenarios = pd.read_csv('cmip5/scenario_names_train.csv').name.to_list()
    lulc_scenarios = pd.read_csv('lulc/scenario_names.csv').name.to_list()
    storage_cost_tree = np.zeros(67*36)
    flood_cost_tree = np.zeros(67*36)
    ####################################################################################################
    ################################ Read indicator files ################################
    ####################################################################################################

    indicators = ['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50','dem_5', 'dem_10','storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10']
    data_ind = {}
    for indicator in indicators:
        data_ind[indicator] = pd.read_csv('indicator_files/%s_indicators.csv' % indicator)
            #indicators are saved from year 2000 similarlr df_Q and demand data are selected

    num = 0
    for gridi in range(67): #ranges utp 97
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
                if policytree == 'donothing':
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

    if P.get_depth() ==1:
      return [10000000000000000000, 1000000000000000000000]
    else:
      return [storage_cost_tree.mean(), flood_cost_tree.mean()]#np.percentile(storage_cost_tree, 99), np.percentile(flood_cost_tree, 99)
    #this returns the objectives of policy tree mean across scenarios
    #return [storage_cost_tree.mean(), flood_cost_tree.mean()]

algorithm = PTreeOpt(wrapper_opt,
                     feature_bounds=[[25,248],[65,181],[75, 163], [2400, 5098], [2717, 4794],[3215, 4309], [365.2, 628], [365.2, 628], [0, 10000],[0, 10000],[0, 10000],[0, 10000]],
                     feature_names=['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50', 'dem_5', 'dem_10', 'storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10'],
                     discrete_actions=True,
                     action_names=['Reopt_5', 'Reopt_10','Reopt_20','Reopt_50','donothing'],
                     mu=20,multiobj= True, epsilons= [1000, 1000],
                     cx_prob=0.70,
                     population_size=100,
                     max_depth=5)

#feature_names=['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50', 'dem_5', 'dem_10', 'storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10'],
#feature_bounds=[[25,248],[65,181],[75, 163], [2400, 5098], [2717, 4794],[3215, 4309], [365.2, 628], [365.2, 628], [10000, 100000],[10000, 100000],[10000, 100000],[10000, 100000]],
                    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s') 

    for seed in range(8,10):
      np.random.seed()

#      with MPIExecutor() as executor:
#          best_solution, best_score, snapshots = algorithm.run(max_nfe=1000,
#                                                 log_frequency=100,
#                                                 snapshot_frequency=100,
#                                                 executor=executor)
      with MultiprocessingExecutor(processes=256) as executor:
        best_solution, best_score, snapshots = algorithm.run(max_nfe=1000,
                                                 log_frequency=100,
                                                 snapshot_frequency=100,
                                                 executor=executor)                                                 
      pickle.dump(snapshots, open('snapshots/result_perf_allscen_upind_%d.pkl' %seed, 'wb'))