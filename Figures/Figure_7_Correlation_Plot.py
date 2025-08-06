import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import seaborn as sns
from shap import summary_plot
sys.path.append('..')
import os
sys.path.append(os.path.abspath("../Main_optimization"))
import Main_optimization.ptreeopt
from Main_optimization.ptreeopt import PTreeOpt
import pickle
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import accuracy_score
import xgboost
from IPython.display import HTML

if __name__ == '__main__':
    ind_list = ['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50','dem_5', 'dem_10','storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10']
    act_list = ['Reopt_5', 'Reopt_10', 'Reopt_20', 'Reopt_50', 'donothing']
    mapping = {'Reopt_5': 0, 'Reopt_10': 1, 'Reopt_20': 2, 'Reopt_50': 3, 'donothing': 4}
    num=0
    c= pd.DataFrame()
    d_all = []

    for seed in range(0, 1):
      for i in range(15): #[2,14,6]: #range(15)
        print(i)
        b = pd.read_csv('b_%d.csv' %(i)) #These files are generated from Figure3_results code
        c1 = pd.read_csv('c_%d.csv' %(i), index_col=0)
        print(np.shape(c1))
        c = pd.concat([c, c1])
        d1 = []

        d = json.load(open('d_%d.json' %(i)))
        for j in range(len(b.values)):
            d1.append(d[str(b.iloc[j, 0])])

        d_all = d_all + d1
    #c.to_csv('c_all.csv')
    #np.savetxt('d_all.csv', d_all) # solve the issue of saving this file
    act=[]
    #map d_all to numbers based on actions
    for j in range(len(d_all)):
        act.append(mapping[d_all[j]])
    with open('c','wb') as fp:
        pickle.dump(c, fp)
    with open('act', 'wb') as fp:
        pickle.dump(act, fp)
    with open('act', 'rb') as fp:
        act_all = pickle.load(fp)
    with open('c', 'rb') as fp:
        c_all = pickle.load(fp)

    X_train,X_test,Y_train,Y_test = train_test_split(c_all,act_all, test_size=0.2, random_state=0)
    shap.initjs()
    model = xgboost.XGBClassifier(objective="multi:softprob", max_depth=4, n_estimators=10,use_label_encoder =False, num_class = 5)#binary:logistic"
    model.fit(X_train, Y_train)

    explainer = shap.TreeExplainer(model)
    shap_values1 = explainer(X_train)
    shap_values = shap.TreeExplainer(model).shap_values(X_train)

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax = ax.ravel()
    data_corr = np.zeros((np.shape(shap_values)[0],np.shape(shap_values)[1]))
    for i in range(np.shape(shap_values)[0]):
        for j in range(np.shape(shap_values)[1]):
            data_corr[i, j] = shap_values[i, j, 3]
    data_all = pd.DataFrame(data_corr, columns=ind_list)
    sns.heatmap(data_all.corr(), ax=ax[0], vmin=-0.1, vmax=1, cmap='RdYlBu',xticklabels=ind_list,yticklabels=ind_list,square=True)
    ####################################################################
    ###################################################################
    # #plot corrrelation
    data_all = pd.DataFrame(shap_values1.data, columns=ind_list)
    sns.heatmap(data_all.corr(),ax=ax[1], vmin=-0.1, vmax=1, cmap='RdYlBu',xticklabels=ind_list,yticklabels=ind_list,square=True)
    ax[1].collections[0].colorbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig('corr.svg')


