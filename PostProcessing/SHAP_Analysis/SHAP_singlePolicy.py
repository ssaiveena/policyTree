import numpy as np
import pandas as pd
import sys
from shap import summary_plot
import matplotlib.pyplot as plt
sys.path.append('..')
from ptreeopt import PTreeOpt
import json
from sklearn.model_selection import train_test_split
import shap
import xgboost

if __name__ == '__main__':
    ind_list = ['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50','dem_5', 'dem_10','storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10']
    act_list = ['Reopt_5', 'Reopt_10', 'Reopt_20', 'Reopt_50', 'donothing']

    act_triggers = pd.read_csv('act_triggers.csv') #b has actions triggered for the entire dataset #actions should be predicted
    SHAP_data = pd.read_csv('SHAP_data.csv', index_col=0) #data with indicators and its values #actions prediction based on the indicator threshold values
    act = list(json.load(open('act.json')).values()) #mapping number to actions triggered for the specific policy tree


    X_train,X_test,Y_train,Y_test = train_test_split(SHAP_data,act_triggers, test_size=0.2, random_state=0)  #train_test_split(x,y)
    shap.initjs()
    model = xgboost.XGBClassifier(objective="multi:softprob", max_depth=4, n_estimators=10,use_label_encoder =False, num_class = 3)#define classifier model
    model.fit(X_train, Y_train) #fitting of the model

    # compute the SHAP values for the model
    explainer = shap.TreeExplainer(model)
    shap_values = shap.TreeExplainer(model).shap_values(X_train)

    mean_0 = np.mean(np.abs(shap_values[0]),axis=0)
    mean_1 = np.mean(np.abs(shap_values[1]),axis=0)
    mean_2 = np.mean(np.abs(shap_values[2]),axis=0)


    df = pd.DataFrame({act[0]:mean_0,act[1]:mean_1,act[2]:mean_2})
    # plot mean SHAP values
    fig,ax = plt.subplots(1,1,figsize=(7, 5))
    df.T.plot.bar(ax=ax)
    ax.set_ylabel('Mean SHAP',size = 14)
    ax.set_xticklabels(act,rotation=45,size=14)
    ax.legend(X_train.columns, fontsize=14)
    plt.tight_layout()
    plt.show()

