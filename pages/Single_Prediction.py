#################### 
# wird nicht mehr benötigt, da in Main.py eingebaut
####################



import streamlit as st
from streamlit_shap import st_shap
import pickle
import pandas as pd
import sys
sys.path.append('../')
import train.config as config
from train.PredModel import PredModel
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from plotly.subplots import make_subplots
import sys
import shap
import keras
import math
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os.path
from functions import *
import features as train_features
import matplotlib.pyplot as plt

####################
# Load Data and Models
####################

X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test
model_dict = st.session_state.model_dict 
model = st.session_state.model
loaded_model = st.session_state.loaded_model
mlalgorithm = st.session_state.mlalgorithm
mlframework = st.session_state.mlframework
logged_model = st.session_state.logged_model

##### Load Original Model #####
if ('sklearn' in str(loaded_model)):
    orig_model = mlflow.sklearn.load_model(logged_model)
if ('tensorflow' in str(loaded_model)):
    import mlflow.keras
    orig_model = mlflow.keras.load_model(logged_model)

st.header("Get Single Cutoffenergy Prediction")

st.subheader("Parameter selector") 



data = config.data


####################
# Parameter Selector
####################

max_energy = float(400) #140
max_pulsewidth = float(data.quantile(q=0.95, axis=0, numeric_only=True)['pulsewidth'])
max_spotsize = float(12)
max_targetthickness = float(data.quantile(q=0.9, axis=0, numeric_only=True)['targetthickness']) 
col1, col2 = st.columns(2)
with col1:
    x_energy = st.slider("Energy (J)",min_value=float(config.data.min(numeric_only=True)['energy'].item()), max_value=max_energy)
    x_spotsize = st.slider("Spot size (µm)", float(config.data.min(numeric_only=True)['spotsize']) , max_spotsize)#float(config.data.quantile(q=0.99, axis=0, numeric_only=True)['spotsize']))
    on_orig = st.toggle('display original data points')
with col2:
    x_pulsewidth = st.slider("Pulse width (fs)",float(config.data.min(numeric_only=True)['pulsewidth']) ,max_pulsewidth)
    x_targetthickness = st.slider("Target thickness (nm)",float(config.data.min(numeric_only=True)['targetthickness']) ,max_targetthickness)

    df_single_pred = pd.DataFrame({ 
                        'energy' : [x_energy], 
                        'pulsewidth' : [x_pulsewidth],
                        'spotsize' : [x_spotsize],
                        'targetthickness' : [x_targetthickness],
                        })
    
    y = get_prediction(df_single_pred, mlalgorithm, loaded_model)
    st.write(y[0])

st.write(mlalgorithm)
####################
# Sidebar
####################
with st.sidebar:     
    mlalgorithm = st.session_state.mlalgorithm
    col1, col2 = st.columns(2)
    if "stacking-monotone" in mlalgorithm:
        y_pred = get_2d_predictions(mlalgorithm, X_test, loaded_model, 'energy', test=True)
        
        col2.metric('Current RMSE', format(mean_squared_error(y_test, y_pred, squared=False), '.2f'))
        col1.metric('Model RMSE', format(model_dict[model]['run']['metrics.RMSE'], '.2f'))
    else:
        col1.metric('Model RMSE', format(model_dict[model]['run']['metrics.RMSE'], '.3f'))
        y_pred = get_2d_predictions(mlalgorithm, train_features.X_test_raw, loaded_model, 'energy', test=True)
        col2.metric('RawTestSet RMSE', format(mean_squared_error(train_features.y_test_raw, y_pred, squared=False), '.3f'))
        y_pred = get_2d_predictions(mlalgorithm, train_features.X_test_wo_outliers, loaded_model, 'energy', test=True)
        st.metric('WoOutlierTestSet RMSE', format(mean_squared_error(train_features.y_test_wo_outliers, y_pred, squared=False), '.3f'))
st.write('test')
####################
# Shap
####################
###### SHAP #####
fig = plt.figure()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()
# Create a SHAP explainer object
if 'flaml' in str.lower(mlframework):
    explainer = shap.TreeExplainer(orig_model.model.estimator, X_train)
if 'tensorflow' in str.lower(mlframework) and 'TFLattice' not in mlalgorithm:
    explainer = shap.KernelExplainer(orig_model.predict, X_train)
if 'TFLattice' in mlalgorithm:
    import tensorflow as tf
    def new_predict_fn(x):
        # Split the input tensor into 4 separate tensors
        inputs = tf.split(x, num_or_size_splits=4, axis=1)
        return loaded_model.predict(inputs)
    data_df = pd.concat([X_test['energy'], X_test['pulsewidth'], X_test['spotsize'], X_test['targetthickness']], axis=1)
    explainer = shap.KernelExplainer(new_predict_fn, data_df)
# Compute SHAP values
# Compute SHAP values
shap_values = explainer(X_test)
#plot shap waterfall plot for the single predicted value y[0]
shap_values = explainer.shap_values(df_single_pred)
st_shap(shap.force_plot(explainer.expected_value, shap_values, df_single_pred)) 
#and now a waterfall plot instead of force plot
st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0]))

# # Create a SHAP explainer object
shap_values = explainer(df_single_pred)
shap_object = shap.Explanation(base_values = shap_values[0][0].base_values,
values = shap_values[0].values,
feature_names = X_test.columns,
data = shap_values[0].data)
shap_values_single = shap_values[0]
st_shap(shap.plots.waterfall(shap_object))
# # Create a summary plot
# st_shap(shap.plots.bar(shap_values))
# st_shap(shap.summary_plot(shap_values, X_test))
# st_shap(shap.dependence_plot("energy", shap_values.values, X_test))
# st_shap(shap.dependence_plot("pulsewidth", shap_values.values, X_test))
# st_shap(shap.dependence_plot("spotsize", shap_values.values, X_test))
# st_shap(shap.dependence_plot("targetthickness", shap_values.values, X_test))