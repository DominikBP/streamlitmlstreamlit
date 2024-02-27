#%%
"""
features.py for train-test split and outlier removal
"""
import os
import sys
import inspect
import config
import pandas as pd
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
pd.options.display.float_format = '{:.3f}'.format # non scientific notation
#%% Load Data
data = config.data 
# slice laser data and add ... to sliced string
data['Laser'] = data['Laser'].apply(lambda x: x[:17] + '...' if len(x) > 17 else x)
data_raw = data.copy()
#%% Remove null values from relevant columns
data = data[config.X]
data = data.loc[:, data.columns != 'angle'].replace(0, np.nan, inplace=False)
data = data.dropna()

#data = data.loc[(data != 0).all(1)] # drop rows with one or more 0 values (use .any insead of .all if only rows with all values 0 should be dropped)
#Alternativ: data.replace(0, np.nan, inplace=True)
#%% 

#%% Detect Outliers using DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mms = MinMaxScaler()
_scaled = data.copy()
if 'intensity' in _scaled.columns:
    _scaled = _scaled.drop('intensity', axis=1)
_scaled[['energy','spotsize','pulsewidth', 'targetthickness', 'cutoffenergy']] = mms.fit_transform(_scaled[['energy','spotsize','pulsewidth', 'targetthickness', 'cutoffenergy']])
_scaled.head()
eps = 0.05
min_samples = 10
model = DBSCAN(eps = eps, min_samples = min_samples).fit(_scaled)
colors = model.labels_
data['labels'] = model.labels_
data['labels'].value_counts()
#%% # Set X,y train-test Split before Outlier Removal
X_Columns = config.X.copy() # copy to avoid changing config.X
X_Columns.append('labels') # add labels to X_Columns
X = data[X_Columns].drop("cutoffenergy", axis=1, inplace=False)
y = data[['cutoffenergy', 'labels']]
seed = 228 # random seed for reproducibility
np.random.seed(seed) # set random seed to np
test_size = 0.2
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=seed)

#%% deprecated
##### remove "outliers" 
#sdata = data.loc[data['energy'] < 50]
# data = data.loc[data['pulsewidth'] < 3000]
# data = data.loc[data['targetthickness'] < 10000]

# deprecated: set upper and lower limit string for mlflow
str_ll = "none"
str_ul = "none"
upper_limit = 0 # 0.85
lower_limit = 0 # 0.00

#%% set Train-/Test data without outliers
X_train_wo_outliers = X_train_raw.loc[X_train_raw['labels'] != -1]
X_test_wo_outliers = X_test_raw.loc[X_test_raw['labels'] != -1]
y_train_wo_outliers = y_train_raw.loc[y_train_raw['labels'] != -1]
y_test_wo_outliers = y_test_raw.loc[y_test_raw['labels'] != -1]
# remove outlier label
X_train_wo_outliers = X_train_wo_outliers.drop('labels', axis=1)
X_test_wo_outliers = X_test_wo_outliers.drop('labels', axis=1)
y_train_wo_outliers = y_train_wo_outliers.drop('labels', axis=1)
y_test_wo_outliers = y_test_wo_outliers.drop('labels', axis=1)
#%% set train-/Test data with outliers
X_train_raw = X_train_raw.drop('labels', axis=1)
X_test_raw = X_test_raw.drop('labels', axis=1)
y_train_raw = y_train_raw.drop('labels', axis=1)
y_test_raw = y_test_raw.drop('labels', axis=1)
#%% set data without outliers
data_wo_outliers = data.loc[data['labels'] != -1]
data_wo_outliers = data_wo_outliers.drop('labels', axis=1)
data = data.drop('labels', axis=1)

#%% save data to excel
describe_filename = f"/home/dominik/Research-Incubator/train/excel/{config.EXPERIMENT_NAME}_DescribeFeatures_filter_{str_ul}-{str_ll}"
dataset_filename = f"/home/dominik/Research-Incubator/train/excel/{config.EXPERIMENT_NAME}_Dataset_filter_{str_ul}-{str_ll}"

data.describe().transpose().to_excel(describe_filename+".xlsx")
data.describe().transpose().to_html(describe_filename+".html")
data.to_excel(dataset_filename+".xlsx")
data.to_excel("/home/dominik/Research-Incubator/train/dataset.xlsx")
# %%
