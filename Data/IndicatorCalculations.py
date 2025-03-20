import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd
import json
import sys
sys.path.append('..')
from numba import njit
import time
import random
import pickle
import os

def indicator_calculation(data_ind, df_Q, sc, indicator):
    if indicator == 'MAF_1_delta':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(1).mean() * cfs_to_tafd
    elif indicator == 'MAF_50_delta':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(50).mean() * cfs_to_tafd
    elif indicator == 'MAF_5_delta':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(5).mean() * cfs_to_tafd
    return data_ind

cmip5_scenarios = pd.read_csv('cmip5/scenario_names.csv').name.to_list()
lulc_scenarios = pd.read_csv('lulc/scenario_names.csv').name.to_list()

# sc = cmip5_scenarios[gridi]
# sl = lulc_scenarios[gridj]
# df_Q = pd.read_csv('cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)
# df_demand = pd.read_csv('lulc/%s.csv.zip' % sl, index_col=0, parse_dates=True)
# df_Q['dowy'] = np.array([water_day_up(d, y) for d, y in zip(df_Q.index.dayofyear, df_Q.index.year)])

# cfs_to_tafd = 2.29568411 * 10 ** -5 * 86400 / 1000

# indicators = ['MAF_1_delta', 'MAF_50_delta', 'MAF_5_delta']
# for indicator in indicators:
#     data_ind = pd.DataFrame()
#     for sc in cmip5_scenarios:
#         df_Q = pd.read_csv('cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)['ORO_inflow_cfs']

#         data_ind = indicator_calculation(data_ind, df_Q, sc, indicator)
#         data_ind = data_ind.loc['2000-10-01':]
#     data_ind.to_csv('indicator_files/%s_indicators.csv' % indicator)

'''Plotting for supplement'''
indicators = ['MAF_1_delta', 'MAF_50_delta']#, 'MAF_5_delta']
for indicator in indicators:
    data_ind = pd.read_csv('indicator_files/%s_indicators.csv' % indicator)
    # print(data_ind)
    
    chunk_size = 5
    mean_df = data_ind.groupby(np.arange(len(data_ind)) // chunk_size).mean()
    # print(mean_df)
    for i in range(97):
        if indicator == 'MAF_1_delta':
            plt.plot(mean_df.iloc[:,i], color='grey')
        else:
            plt.plot(mean_df.iloc[:,i], color='tab:blue')
plt.xticks([0,5,10,15])
plt.show()