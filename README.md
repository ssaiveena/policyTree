## PolicyTreeOptimization
This repository contains all code corresponding to methods and figure generation in the paper below:

Adaptation triggers and indicator interpretability for dynamic reoptimization of reservoir control policies under climate change

#### Requirements
[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Scipy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [Seaborn](https://seaborn.pydata.org/), [Numba](https://numba.pydata.org/),  [multiprocessing](https://docs.python.org/3/library/multiprocessing.html), [ptreeopt](https://github.com/jdherman/ptreeopt),[pickle](https://docs.python.org/3/library/pickle.html)

### Directories

``Data``: Contains input data for analysis used in the study. 

``Main_optimization``: This folder contains function codes called by ``main_reopt_perf.py`` and supporting function files. This includes the framework for an “outer loop” adaptation policy that establishes indicator thresholds for reoptimization based on recently observed data, and an “inner loop” control policy that undergoes reoptimization according to these thresholds. This directory also has ``main_reopt_perf_reevaluate.py`` that is used to reevaluate teh policies for the training and testing set to determine the policy actions triggered over time based on feature variables describing changes in hydrology and demand.

``Figures``: Directory containing python scripts to generate Figures 3-9 of the manuscript and data used plotting the figures.

``PostProcessing``: Directory containing python scripts on SHAP analysis and sensitivity analysis.

### Data preparation and model run
* The scenario data can be downloaded [here](https://www.dropbox.com/s/gmgujninm02l0e8/scenario_data.zip?dl=1). Unzip and move the folders into `data/cmip5` and `data/lulc`. 
  - The CMIP5 climate scenarios are from [USBR](https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html#About) and contain daily reservoir inflows in cfs. 
  - The LULC scenarios are from multiple models and have been converted to water demand multipliers as described in [Cohen et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021WR030433)
* The solutions of the reoptimization (Reopt_50 in the manuscript) can be obtained from this [repository](https://github.com/ssaiveena/Continuous-Reoptimization)

#### License: [MIT](LICENSE.md)
