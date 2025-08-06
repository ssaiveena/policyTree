import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from os import listdir
from os.path import isfile, join
import seaborn as sns
import pickle
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
sys.path.append(os.path.abspath("../Main_optimization"))
import ptreeopt


snapshots = pickle.load(open('../Main_optimization/snapshots/result.pkl', 'rb'), encoding='latin1')
f = snapshots['best_f'][-1]
P = snapshots['best_P'][-1]
#P = snapshots['best_P'][0]

###################################################################################################################
################Plot indicators ############################################################
###################################################################################################################
# cmip5_scenarios = pd.read_csv('cmip5/scenario_names.csv').name.to_list()
# lulc_scenarios = pd.read_csv('lulc/scenario_names.csv').name.to_list()


# cmip5_w = cmip5_scenarios[14]
# sl = lulc_scenarios[3]

# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 4))
# sns.set_style('darkgrid')

# indicators = ['Peak_Flow_1', 'dem_10', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50',  'dem_5']
# data_ind = {}
# for indicator in indicators:
#     data_ind[indicator] = pd.read_csv('indicator_files/%s_indicators.csv' % indicator)

# data = pd.read_csv('flood_5.txt', header=None)
# data1 = pd.read_csv('storage_10.txt', header=None)

# colors_list = ['tab:blue', 'tab:green', 'tab:red', 'tab:olive']

# col_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]



# label_list = ['donothing','Reopt_50']

# mapping = {0 : '#b2df8a', 2: '#1f78b4', 1: '#e6ab02',3:'#386cb0'}



# axs[0].plot(data_ind['dem_5'][sl], linewidth=2)
# axs[0].hlines(y = 400, xmin = 0, xmax = 98, color = 'r', linestyle = '-',linewidth=2)
# axs[0].set_ylabel('dem_5', fontsize=14)
# axs[0].set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
# axs[0].set_xlim([0,98])
# axs[0].tick_params(axis='both', labelsize=10)
# #
# axs[1].plot(data,linewidth=2)
# axs[1].hlines(y = 6612, xmin = 0, xmax = 98, color = 'r', linestyle = '-',linewidth=2)
# axs[1].set_ylabel('flood_cost_5', fontsize=14)
# axs[1].set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
# axs[1].tick_params(axis='both', labelsize=10)
# axs[1].set_xlim([0,98])


# #axs[0].plot(data_ind['Peak_Flow_10'][cmip5_w], linewidth=2)
# #axs[0].hlines(y = 150, xmin = 0, xmax = 98, color = 'r', linestyle = '-',linewidth=2)
# #axs[0].set_ylabel('Peak_Flow_10', fontsize=14)
# #axs[0].set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
# #axs[0].set_xlim([0,98])
# #axs[0].tick_params(axis='both', labelsize=10)
# #
# #axs[1].plot(data_ind['MAF_5'][cmip5_w],linewidth=2)
# #axs[1].hlines(y = 3213, xmin = 0, xmax = 98, color = 'r', linestyle = '-',linewidth=2)
# #axs[1].set_ylabel('MAF_5', fontsize=14)
# #axs[1].set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
# #axs[1].tick_params(axis='both', labelsize=10)
# #axs[1].set_xlim([0,98])

# for i in range(0, 98):
#     axs[0].axvspan(i-0.5, i+0.5,facecolor=mapping[col_list[i]], alpha=0.8)
#     axs[1].axvspan(i-0.5, i+0.5,facecolor=mapping[col_list[i]], alpha=0.8)
# #    axs[2].axvspan(i-0.5, i+0.5,facecolor=mapping[col_list[i]], alpha=0.5)
# #    axs[3].axvspan(i-0.5, i+0.5,facecolor=mapping[col_list[i]], alpha=0.5)

# red_patch1 = mpatches.Patch(color=mapping[0], label=label_list[0])
# red_patch2 = mpatches.Patch(color=mapping[1], label=label_list[1])
# plt.legend(handles=[red_patch1, red_patch2], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
# plt.tight_layout()
# plt.show()
#################################################################################################################################################################
#############################################################################################################################################################
##plot shading in the background probabilities
###########################################
from matplotlib.gridspec import GridSpec

fig = plt.figure()

gs = GridSpec(4, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[-1, 0])
ax5 = fig.add_subplot(gs[-1, 1])

cmip5_scenarios = pd.read_csv('../Data/cmip5/scenario_names.csv').name.to_list()
lulc_scenarios = pd.read_csv('../Data/lulc/scenario_names.csv').name.to_list()

indicators = ['Peak_Flow_1', 'dem_10', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50',  'dem_5']
data_ind = {}
for indicator in indicators:
    data_ind[indicator] = pd.read_csv('../Data/indicator_files/%s_indicators.csv' % indicator, parse_dates=True)

data_prob_flood = pd.read_csv('probdata_floodcost_FC.txt', header=None, delimiter=' ')

num=0
for gridi in range(97):
    for gridj in range(36):

        # sort the data in ascending order 
        x = np.sort(data_prob_flood.iloc[num, :]) 
        sl = lulc_scenarios[gridj]
        sc = cmip5_scenarios[gridi]          
        # get the cdf values of y 
        y = np.arange(99) / float(99) 
        ax4.plot(x, 1-y, color='k', alpha=0.3) 
        #ax3.plot( np.sort(data_ind['Peak_Flow_10'][sc]), 1-y, color='k', alpha=0.3)
        
        ax5.plot( np.sort(data_ind['dem_5'][sl]), 1-y, color='k', alpha=0.3)
        num=num+1

# plotting 
ax4.set_xlabel('Flood Cost') 
ax5.set_xlabel('Demand') 

ax4.set_ylabel('Probability\nof Exceedance')   

colors_list = ['tab:blue', 'tab:green', 'tab:red', 'tab:olive']

import random
col_list = []
for i in range(0,98):
    n = random.randint(1,30)
    col_list.append(n)

data_prob = pd.read_csv('probdata_FC.text', header=None, delimiter=' ')
data_donothing = (data_prob == 0).astype(int).sum(axis=0)/3492 #as 3492 scenarios
data_reopt = (data_prob == 1).astype(int).sum(axis=0)/3492 #as 3492 scenarios
data_reopt1 = (data_prob == 2).astype(int).sum(axis=0)/3492 #as 3492 scenarios


# col_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

label_list = ['donothing','Reopt_20','Reopt_50']

mapping = {0 : '#b2df8a', 2: '#1f78b4', 1: '#e6ab02',3:'#386cb0'}

# axs[0].plot(data_ind['dem_5'][sl], linewidth=2)
# axs[0].hlines(y = 400, xmin = 0, xmax = 98, color = 'r', linestyle = '-',linewidth=2)
# axs[0].set_ylabel('dem_5', fontsize=14)
# axs[0].set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
# axs[0].set_xlim([0,98])
# axs[0].tick_params(axis='both', labelsize=10)
# #
# axs[1].plot(data,linewidth=2)
# axs[1].hlines(y = 6612, xmin = 0, xmax = 98, color = 'r', linestyle = '-',linewidth=2)
# axs[1].set_ylabel('flood_cost_5', fontsize=14)
# axs[1].tick_params(axis='both', labelsize=10)
# axs[1].set_xlim([0,98])

# Create a Normalize instance to scale the data values to the range [0, 1]
norm = Normalize(vmin=0, vmax=1)

# Create a ScalarMappable for the colormap
cmap = plt.get_cmap('Blues')
scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)

# Now you can use scalar_mappable.to_rgba(value) to get the RGBA value for a specific data value

# Example usage:
for i in range(0, 98):
    ax1.axvspan(i - 0.5, i + 0.5, facecolor=scalar_mappable.to_rgba(data_donothing[i]))
    ax2.axvspan(i - 0.5, i + 0.5, facecolor=scalar_mappable.to_rgba(data_reopt[i]))
    ax3.axvspan(i - 0.5, i + 0.5, facecolor=scalar_mappable.to_rgba(data_reopt1[i]))
    
# Add a colorbar
plt.colorbar(scalar_mappable, ax=[ax1, ax2])

ax1.set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
ax2.set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
ax3.set_xticklabels([2000,2020,2040,2060,2080], fontsize=10)
ax1.set_xlim([0,98])
ax2.set_xlim([0,98])
ax2.set_xlim([0,98])

ax1.set_title('No Action', fontsize=12)
ax2.set_title('Reopt_20', fontsize=12)
ax3.set_title('Reopt_50', fontsize=12)


ax4.vlines(x = 7578, ymin = 0, ymax = 1, color = 'r', linestyle = '-',linewidth=2)
ax5.vlines(x = 594, ymin = 0, ymax = 1, color = 'r', linestyle = '-',linewidth=2)
plt.tight_layout()
plt.savefig('FigureFC.svg')
