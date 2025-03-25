import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
import random
import json
import os
from os import listdir
from os.path import isfile, join
import seaborn as sns
import pickle
from matplotlib import cm
import ptreeopt
from ptreeopt.plotting import *

import pareto

pf_train = [378365, 48684]
pf_test = [395873, 95767]

sim_train = [441680, 80025]
sim_test = [450501, 166320]

###############################################################################################################
######################PLOT TRAINING AND TESTING SCENARIOS#################################################################
# #################################################################################################################
fig, ax = plt.subplots()
arr = [[0 for i in range(2)] for j in range(185)]
arr_test = [[0 for i in range(2)] for j in range(185)]
P_set = []
j=0
for seed in range(9):
    snapshots = pickle.load(open('result_perf_allscen_upind_%d.pkl' % seed, 'rb'), encoding='latin1')
    data = pd.read_csv('result_perf_allscen_upind_%d.txt' % seed, header=None, delimiter=' ')
    f = snapshots['best_f'][-1]
    P = snapshots['best_P'][-1]
    rows = len(P)
    print(rows)
    for i in range(len(P)):
        ax.scatter(f[i][0], f[i][1], color = 'grey',  label='WO Demand')
        arr[j][0], arr[j][1] = f[i][0], f[i][1]
        ax.scatter(data.iloc[i,0], data.iloc[i,1], color='red', marker='^', label='Simulated')
        arr_test[j][0], arr_test[j][1] = data.iloc[i,0], data.iloc[i,1]
        P_set.append(P[i])
        #graphviz_export(P[i], 'test_%d.svg' % i)  # creates one SVG

        j = j + 1

plt.plot(pf_train[0], pf_train[1], color='grey', marker='s')
#plt.plot(pf_test[0], pf_test[1], color='red', marker='s')

plt.xlabel('Storage cost', fontsize=16)
plt.ylabel('Flood cost', fontsize=16)
# ax.scatter(sim_train[0], sim_train[1], color = 'grey', marker='*', s=200, label = 'Simulated')
# ax.scatter(sim_test[0], sim_test[1], color = 'red', marker='*', s=200, label = 'Simulated')

np.savetxt('result_perf.txt', arr)
np.savetxt('result_perf_test.txt', arr_test)
#plt.savefig('res.jpg')
plt.show()

##########################################################################################################################
###################################PLOT SORTED SOLUTIONS##############################################################
#########################################################################################################################
df = pd.read_csv('result_perf.txt', header= None, delimiter=' ')
df_test = pd.read_csv('result_perf_test.txt', header= None, delimiter=' ')
print(df_test)
# # define the convergence tolerance for the OF's
# # optional. default is 1e-9
eps_tols = [1000, 10000]
#eps_tols = [0.005, 0.05]

of_cols=[0,1]
# sort
nondominated = pareto.eps_sort([list(df.itertuples(False))], of_cols, eps_tols)

# convert multi-dimension array to DataFrame
df_pareto = pd.DataFrame.from_records(nondominated, columns=list(df.columns.values))
df_pareto.to_csv('result_pareto_all.txt')

fig, ax = plt.subplots()
plt.plot(pf_train[0], pf_train[1], color='grey', marker='s')
ax.scatter(df_pareto.iloc[:, 0], df_pareto.iloc[:, 1], color='grey')

ax.scatter(df_pareto.iloc[[6,2,14], 0], df_pareto.iloc[[6,2,14], 1], color='grey', edgecolors='black', linewidth=3)

ax.scatter(sim_train[0], sim_train[1], color = 'grey', marker='*', s=200, label = 'Simulated')
#ax.scatter(sim_test[0], sim_test[1], color = 'red', marker='*', s=200, label = 'Simulated')

plt.plot(pf_test[0], pf_test[1], color='red', marker='s')

a  = df.index[df.iloc[:,0].isin(df_pareto.iloc[:,0])].tolist()

plt.xlabel('Shortage cost', fontsize=16)
plt.ylabel('Flood cost', fontsize=16)
# ax.scatter(df_test.iloc[a, 0], df_test.iloc[a, 1], color='red', marker='^')
# ax.scatter(df_test.iloc[[a[6],a[2],a[14]], 0], df_test.iloc[[a[6],a[2],a[14]], 1], color='red', marker='^', edgecolors='black', linewidth=2.5)

# num=0
# for i in range(15):
#     ax.text(df_test.iloc[a[i], 0], df_test.iloc[a[i], 1], str(i), size=15, horizontalalignment='right',
#         verticalalignment='bottom')
#     ax.text(df_pareto.iloc[i, 0], df_pareto.iloc[i, 1], str(i), size=15, horizontalalignment='right',
#         verticalalignment='bottom')
#
#     print(np.sqrt((df_pareto.iloc[i,0])**2 + (df_pareto.iloc[i,1])**2))
#     num=num+1
    # ax.text(df_test.iloc[i, 0], df_test.iloc[i, 1], str(i), size=15, horizontalalignment='right',
    #     verticalalignment='bottom')

plt.savefig('test.svg')
P_sel=[]
for ind in range(len(a)):
    print(P_set[a[ind]])
    P_sel.append(P_set[a[ind]])
    print(df_pareto.iloc[ind,:])
    graphviz_export(P_set[a[ind]], 'test_%d.svg' %ind) # creates one SVG
pickle.dump(P_sel, open('result_SelP.pkl', 'wb'))
###############################################################################################################
###############################################################################################################
###############################################################################################################