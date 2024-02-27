"""
Config file for training
Merge files and clean features, convert units..
"""
import os
import pandas as pd

#%% get experiment name
file = open("/home/dominik/Research-Incubator/train/config_exp_name.txt", "r")
EXPERIMENT_NAME = file.read()
print (EXPERIMENT_NAME)
#%% get data from data files
datafile1 = os.path.join(os.path.dirname(os.getcwd()), 'data/data_incl_source.xlsx')
datafile2 = os.path.join(os.path.dirname(os.getcwd()), 'data/Datensammlung_update.xlsx')

data1 = pd.read_excel(datafile1)
data2 = pd.read_excel(datafile2)
data1_raw = data1

#%%

columnscsv = [
          'Energy J', 
          'pulse width', 
          'Cut of energy', 
          'spot size (d)', 
          'Targetthickness', 
          'Intensity', 
          'Laser',
          'Power',
          'angle',
          'Vary'
    ]
data1 = data1[columnscsv]
data1 = data1.rename(columns={
          columnscsv[0]:"energy",
          columnscsv[1]:"pulsewidth",
          columnscsv[2]:"cutoffenergy",
          columnscsv[3]:"spotsize",
          columnscsv[4]:"targetthickness", 
          columnscsv[5]:"intensity",
          #columnscsv[6]:"laser", #hat schon richtigen namen?
          columnscsv[7]:"power",
          columnscsv[8]:"angle",
          columnscsv[9]:"vary"
    })


#%%
data1.describe()
#data1.tail()
#%%
# Datensammlung
columns2 = [
          'Energy (in Joule)', 
          'pulse width (in fs)', 
          'Cut of energy', 
          'spot size (d)', 
          'Targetthickness (in µm)', 
          'Intensity', 
          'Laser',
          'Power',
          'angle',
          'Vary',
          'Target Material'

    ]
data2 = data2[columns2]
data2 = data2.rename(columns={
          "Energy (in Joule)":"energy", 
          "pulse width (in fs)":"pulsewidth",
          "Cut of energy":"cutoffenergy",	
          "spot size (d)":"spotsize", 
          "Targetthickness (in µm)":"targetthickness", 
          "Intensity":"intensity",
          #"Laser":"laser",
          "Power":"power",
          "Vary":"vary",
          "Target Material":"targetmaterial",

    })

#%% Targetthickness from micrometer to nanometer (*1000)
data2.targetthickness = data2.targetthickness*1000
#%%
data_raw = pd.concat([data1, data2])
data_raw.head()
#%%
data_raw['Laser'] = data_raw['Laser'].replace('PHELIX', 'Phelix')
data = data_raw

# %% Set X for MLflow Logging in model train runs
X = ['cutoffenergy', 'energy', 'pulsewidth', 'spotsize', 'targetthickness']
y = ['cutoffenergy']

# %%
