import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
with open('scenario_names_lulc.txt') as f:
    scenarios = f.read().splitlines()

cfs_to_tafd = 2.29568411 * 10 ** -5 * 86400 / 1000
def indicator_calculation(data_ind, df_Q, sc, indicator):
    if indicator == 'Peak_Flow_1':
        data_ind[sc] = df_Q.resample('AS-OCT').max().rolling(1).mean() * cfs_to_tafd
    elif indicator == 'Peak_Flow_5':
        data_ind[sc] = df_Q.resample('AS-OCT').max().rolling(5).mean() * cfs_to_tafd
    elif indicator == 'Peak_Flow_10':
        data_ind[sc] = df_Q.resample('AS-OCT').max().rolling(10).mean() * cfs_to_tafd
    elif indicator == 'MAF_5':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(5).mean() * cfs_to_tafd
    elif indicator == 'MAF_10':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(10).mean() * cfs_to_tafd
    elif indicator == 'MAF_50':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(50).mean() * cfs_to_tafd
    elif indicator == 'dem_5':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(5).mean()
    elif indicator == 'dem_10':
        data_ind[sc] = df_Q.resample('AS-OCT').sum().rolling(10).mean()
    return data_ind

indicators = ['dem_5', 'dem_10']
# ['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50',
# for indicator in indicators:
#     data_ind = pd.DataFrame()
#     for sc in scenarios:
#         df_Q = pd.read_csv('lulc/%s.csv.zip' % sc, index_col=0, parse_dates=True)['combined_demand']
#         data_ind = indicator_calculation(data_ind, df_Q, sc, indicator)
#         data_ind = data_ind.loc['2000-10-01':]
#     data_ind.to_csv('indicator_files/%s_indicators.csv' % indicator)

# '''finding the 90th and 10th percentile values of indicators to input as bounds'''
# for indicator in indicators:
#     df_Q = pd.read_csv('indicator_files/%s_indicators.csv' % indicator)
#     print(np.max(df_Q)[1:].max())
#     print(np.min(df_Q)[1:].min())
# from numpy import random
#
# x = random.randint(100, size=(100))
# print(x)
# print(np.mean(x[0:5]))
# print(np.mean(x[5:10]))
############################################################################################################################
###############################################################################################################################
#######################Plot frequency of idicators########################################################################
act = ['a) donothing','b) Reopt_5', 'c) Reopt_10','d) Reopt_20','e) Reopt_50','f) Reopt']
ind_list = ['Peak_Flow_1', 'Peak_Flow_5', 'Peak_Flow_10', 'MAF_5', 'MAF_10', 'MAF_50', 'dem_5', 'dem_10', 'storage_cost_5', 'flood_cost_5', 'storage_cost_10', 'flood_cost_10']
# act1 = [5945, 10058, 27887, 23748, 17084, 33194, 50784, 13653, 8381, 31387, 1455, 51619]
# act2 = [3561, 4744, 7185, 9287, 2588, 14057, 30677, 12382, 3984, 13362, 8089, 5617]
# act3 = [0, 552, 0, 0, 3717, 3050, 0, 1525, 0, 2744, 0, 552]
# act4 = [0, 0, 179, 0, 172, 1147, 20, 1054, 0, 29, 86, 23]
# act5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

act1 = [224784, 449353, 462804, 407196, 505044, 539894, 590353, 383636, 443868, 396970, 41808, 464248]
act2 = [11412, 76822, 101196, 74160, 480, 55474, 150806, 21348, 11412, 95368, 480, 21348]
act3 = [0, 7263, 0, 0, 67608, 14526, 0, 7263, 0, 67608, 0, 7263]
act4 = [0, 0, 0, 0, 32, 1918, 243, 959, 0, 32, 991, 211]
act5 = [180, 62174, 49764, 49536, 93204, 231571, 327006, 59546, 32952, 176186, 195087, 0]

act6= [sum(x) for x in zip(act2, act3,act4,act5)]

font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))

df=pd.DataFrame({'allvarlist':ind_list,'importances':[x/sum(act1) for x in act1] })
df.sort_values('importances',inplace=True)
df.plot(kind='barh',y='importances',x='allvarlist',color='#0571b0', legend=False,ax = axs[0,0])
axs[0,0].set_title(act[0])
print(df)

df=pd.DataFrame({'allvarlist':ind_list,'importances':[x/sum(act2) for x in act2]})
df.sort_values('importances',inplace=True)
df.plot(kind='barh',y='importances',x='allvarlist',color='#0571b0', legend=False,ax = axs[0,1])
axs[0,1].set_title(act[1])
print(df)

df=pd.DataFrame({'allvarlist':ind_list,'importances':[x/sum(act3) for x in act3]})
df.sort_values('importances',inplace=True)
df.plot(kind='barh',y='importances',x='allvarlist',color='#0571b0', legend=False,ax = axs[0,2])
axs[0,2].set_title(act[2])
print(df)

df=pd.DataFrame({'allvarlist':ind_list,'importances':[x/sum(act4) for x in act4]})
df.sort_values('importances',inplace=True)
df.plot(kind='barh',y='importances',x='allvarlist',color='#0571b0', legend=False,ax = axs[1,0])
axs[1,0].set_title(act[3])
print(df)

df=pd.DataFrame({'allvarlist':ind_list,'importances':[x/sum(act5) for x in act5]})
df.sort_values('importances',inplace=True)
df.plot(kind='barh',y='importances',x='allvarlist',color='#0571b0', legend=False,ax = axs[1,1])
axs[1,1].set_title(act[4])
print(df)

df=pd.DataFrame({'allvarlist':ind_list,'importances':[x/sum(act6) for x in act6]})
df.sort_values('importances',inplace=True)
df.plot(kind='barh',y='importances',x='allvarlist',color='#0571b0', legend=False,ax = axs[1,2])
axs[1,2].set_title(act[5])
print(df)

axs[0,0].set_ylabel('')
axs[0,1].set_ylabel('')
axs[1,0].set_ylabel('')
axs[1,1].set_ylabel('')
axs[0,2].set_ylabel('')
axs[1,2].set_ylabel('')

plt.show()
############################################################################################################################
###############################################################################################################################
#######################Plot short term indicators with time########################################################################
count1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
count2 = [97, 93, 92, 91, 94, 92, 92, 94, 94, 93, 94, 95, 95, 92, 93, 92, 94, 93, 93, 94, 93, 92, 93, 95, 92, 90, 90, 88, 90, 90, 88, 87, 89, 88, 86, 84, 81, 86, 90, 85, 87, 87, 87, 88, 82, 87, 86, 88, 83, 80, 80, 79, 83, 81, 80, 82, 82, 79, 80, 82, 83, 80, 79, 78, 80, 79, 75, 75, 74, 73, 74, 76, 75, 74, 74, 71, 66, 67, 68, 67, 72, 70, 71, 69, 68, 68, 71, 73, 71, 68, 73, 76, 69, 71, 67, 66, 67, 61]
count3 = [71, 73, 77, 75, 77, 76, 80, 79, 83, 78, 77, 76, 73, 75, 74, 77, 77, 81, 84, 78, 82, 80, 82, 80, 83, 86, 81, 85, 84, 85, 84, 82, 79, 79, 86, 87, 86, 88, 87, 87, 83, 87, 88, 85, 89, 89, 89, 89, 89, 89, 87, 89, 90, 89, 90, 88, 88, 89, 86, 87, 89, 88, 84, 79, 84, 87, 88, 91, 90, 90, 90, 90, 90, 90, 88, 87, 89, 90, 89, 90, 93, 93, 94, 95, 91, 91, 89, 90, 89, 88, 90, 90, 91, 91, 93, 92, 91, 93]
count4 = [71, 73, 77, 75, 77, 76, 80, 79, 83, 78, 77, 76, 73, 75, 74, 77, 77, 81, 84, 78, 82, 80, 82, 80, 83, 86, 81, 85, 84, 85, 84, 82, 79, 79, 86, 87, 86, 88, 87, 87, 83, 87, 88, 85, 89, 89, 89, 89, 89, 89, 87, 89, 90, 89, 90, 88, 88, 89, 86, 87, 89, 88, 84, 79, 84, 87, 88, 91, 90, 90, 90, 90, 90, 90, 88, 87, 89, 90, 89, 90, 93, 93, 94, 95, 91, 92, 89, 90, 89, 88, 90, 90, 91, 91, 93, 92, 91, 93]
count5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

plt.plot(count1)
plt.plot(count2)
plt.plot(count3)
plt.plot(count4)
plt.plot(count5)
plt.legend(['Peak_Flow_1','MAF_5','dem_5','storage_cost_5','flood_cost_5'])
plt.show()
