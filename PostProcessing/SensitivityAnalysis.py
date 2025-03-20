import sys
from SALib.analyze import delta
from SALib.util import read_param_file
import numpy as np
import pandas as pd
import json
# from multiprocessing import Pool, current_process
import os
sys.path.append("../..")
#save as 5-years groups
# for gridi in range(97):
#     for gridj in range(36):
#         #input_data = pd.read_csv('input_output_data_PF/inputs_%d_%d.csv' %(gridi, gridj), delimiter=',')
#         output_data = pd.read_csv('input_output_data_PF/rel_%d_%d.csv' %(gridi, gridj), delimiter=',')
#         input_data = pd.Series(output_data.values.ravel()).dropna().reset_index(drop=True)
#         data = input_data#.iloc[:,0]
#         # Generate date range from Oct 1, 2000 to Sep 30, 2098
#         start_date = "2000-10-01"
#         end_date = "2098-09-30"
#         dates = pd.date_range(start=start_date, end=end_date, freq="D")
#         feb_29_indices = [i for i, date in enumerate(dates) if date.month == 2 and date.day == 29]
#         filtered_data = data.drop(index=feb_29_indices)
#         # Remove Feb 29
#         dates = dates[~((dates.month == 2) & (dates.day == 29))]

#         # Create a DataFrame with the data and dates
#         df = pd.DataFrame({"Date": dates[0:34675], "Value": filtered_data[0:34675]})

#         # Assign Water Year (Oct to Sep)
#         df["Water_Year"] = df["Date"].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)

#         # Group into 5-year blocks based on water years
#         df["5yr_Block"] = np.repeat(np.arange(0,19),365*5)

#         # Split data into columns for each 5-year block
#         blocks = []
#         for block_id, group in df.groupby("5yr_Block"):
#             if block_id < 19:
#                 blocks.append(group["Value"].reset_index(drop=True))
#         # Combine into a DataFrame
#         result_df = pd.concat(blocks, axis=1)
#         result_df.columns = [f"Water_Year_Block_{2000 + 5 * i}-{2004 + 5 * i}" for i in range(len(blocks))]
#         if gridi==0 and gridj==0:
#             final_df = result_df
#         else:
#             final_df = pd.concat([final_df, result_df], axis=1)

# for ind in range(1825):
#     np.savetxt('Sensitivity_analysis_PF/output_first_%d.txt' %ind, final_df.iloc[ind,:].values)
# -----------------------------------------------------------------------------------------------------
#save five year data into one group
##%
# for gridi in range(97):
#     for gridj in range(36):
#         input_data = pd.read_csv('input_output_data/inputs_%d_%d.csv' %(gridi, gridj), delimiter=',')
#         # output_data = pd.read_csv('input_output_data/rel_%d_%d.csv' %(gridi, gridj), delimiter=',')
#         # input_data = pd.Series(output_data.values.ravel()).dropna().reset_index(drop=True)
#         data = input_data.iloc[:,2]
#         # Generate date range from Oct 1, 2000 to Sep 30, 2098
#         start_date = "2000-10-01"
#         end_date = "2098-09-30"
#         dates = pd.date_range(start=start_date, end=end_date, freq="D")
#         feb_29_indices = [i for i, date in enumerate(dates) if date.month == 2 and date.day == 29]
#         filtered_data = data.drop(index=feb_29_indices)
#         # Remove Feb 29
#         dates = dates[~((dates.month == 2) & (dates.day == 29))]

#         # Create a DataFrame with the data and dates
#         df = pd.DataFrame({"Date": dates[0:34675], "Value": filtered_data[0:34675]})

#         # Assign Water Year (Oct to Sep)
#         df["Water_Year"] = df["Date"].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)

#         # Group into 5-year blocks based on water years
#         df["5yr_Block"] = np.repeat(np.arange(0,19),365*5)
#         # Split data into columns for each 5-year block
#         blocks = []
#         for block_id, group in df.groupby("5yr_Block"):
#             if block_id < 19:
#                 blocks.append(group["Value"].reset_index(drop=True))
#         # Combine into a DataFrame
#         result_df = pd.concat(blocks, axis=1)
#         result_df.columns = [f"Water_Year_Block_{2000 + 5 * i}-{2004 + 5 * i}" for i in range(len(blocks))]
#         if gridi==0 and gridj==0:
#             final_df = result_df
#         else:
#             final_df = pd.concat([final_df, result_df], axis=0)

# for ind in range(19):
#     np.savetxt('Sensitivity_analysis/thridinput_%d.txt' %ind, final_df.iloc[:,ind].values)
# -----------------------------------------------------------------------------------------------------
#perform sensitivity analysis from here
# #this analysis ignores leap years

problem = {
    'num_vars': 3,
    'names': ['a','b','c'],
    'bounds': [[0,1],[-1,1],[0,1]]
}

def func(i):
    input_data1 = np.loadtxt('Sensitivity_analysis/firstinput_%d.txt' %i)
    input_data2 = np.loadtxt('Sensitivity_analysis/secondinput_%d.txt' %i)
    input_data3 = np.loadtxt('Sensitivity_analysis/thridinput_%d.txt' %i)
    input_data = pd.DataFrame({'Column1': input_data1,
                               'Column2': input_data2, 'Column3': input_data3})
    # the data is from 10/1/2000 to 09/30/2098 with 35794 elemnts
    output_data = np.loadtxt('Sensitivity_analysis/firstoutput_%d.txt' %i)

    result = delta.analyze(problem, input_data.iloc[:,0:3].to_numpy(), output_data, print_to_console=False)#, num_resamples=10)

    results_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in result.items()}
    with open('SAoutput_CP/SAresult_5yr_%d' %i, "w") as json_file:
        json.dump(results_serializable, json_file, indent=4)


# exp_a = range(1825) #list to loop through climate scenarios

# # auxiliary funciton to make it work
# def product_helper(args):
#     return func(args)

# def parallel_product(list_a):
#     #spark given number of processes
#     p = Pool(200)
#     # set each matching item into a tuple
#     job_args = [x for x in list_a]

#     # map to pool
#     p.map(product_helper, job_args)

# if __name__ == '__main__':
#     parallel_product(exp_a)

# Define a range for computation
# n = 1825
# num_chunks = 200
# chunk_size = n // num_chunks

# # Define chunks (start, end) for each process
# chunks = [(i, i + chunk_size) for i in range(0, n, chunk_size)]

# Use multiprocessing Pool
# with Pool(processes=64) as pool:
#     pool.map(func, np.arange(64).tolist())

for i in range(0,3):
    func(i)