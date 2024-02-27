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
import features as train_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os.path
from streamlit_super_slider import st_slider

from functions import *
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test

loaded_model = st.session_state.loaded_model
mlalgorithm = st.session_state.mlalgorithm

st.header("Average Scaling Law Cutofenergy Comparison")


st.subheader("Parameter selector") 

col1, col2 = st.columns(2)

data = config.data

max_energy = round(train_features.data_wo_outliers['energy'].max(), 1)+50
#max_energy = float(250) #140
#max_pulsewidth = round(train_features.data_wo_outliers['pulsewidth'].max(), None)+50
max_pulsewidth = round(float(data.quantile(q=0.95, axis=0, numeric_only=True)['pulsewidth']), 1)
#max_spotsize = round(train_features.data_wo_outliers['spotsize'].max(), None)+50
#max_spotsize = float(12)
max_spotsize = round(float(data.quantile(q=0.95, axis=0, numeric_only=True)['spotsize']), 1)
#max_targetthickness = round(train_features.data_wo_outliers['targetthickness'].max(), None)+50
max_targetthickness = round(float(data.quantile(q=0.8, axis=0, numeric_only=True)['targetthickness']), 1)
###### parameters
# todo: dynamic n_mean, slider_smoother_power, slider_monotone_weight // in sidebar
n_mean = 5
slider_smooth_power = 1
slider_monotone_weight = 0
##### parameter selector
with col1:
    st.markdown(':blue[Energy (J)]')
    x_energy = st_slider(min_value=float(config.data.min(numeric_only=True)['energy'].item()), max_value=max_energy)
    x_energy = 2.4
    st.markdown(':green[Spot size (µm)]')
    x_spotsize = st_slider(min_value=float(config.data.min(numeric_only=True)['spotsize']) , max_value=max_spotsize)#float(config.data.quantile(q=0.99, axis=0, numeric_only=True)['spotsize']))
    x_spotsize = 3.3
    on_orig = st.toggle('Show dataset')
with col2:
    st.markdown(':red[Pulse width (fs)]')
    x_pulsewidth = st_slider(float(config.data.min(numeric_only=True)['pulsewidth']) ,max_pulsewidth)
    x_pulsewidth = 31.3
    st.markdown(':violet[Target thickness (nm)]')
    x_targetthickness = st_slider(float(train_features.data.min(numeric_only=True)['targetthickness']) ,max_targetthickness)
    x_targetthickness = 635

    df_single_pred = pd.DataFrame({ 
                        'energy' : [x_energy], 
                        'pulsewidth' : [x_pulsewidth],
                        'spotsize' : [x_spotsize],
                        'targetthickness' : [x_targetthickness],
                        })
    
    # y = get_prediction(df_single_pred)
    # st.write(str(y.values[0]))

# ToDo: if stacking monotone in mlalgorithm
st.divider()

fig = make_subplots(rows = 2, cols=2, start_cell="top-left",horizontal_spacing = 0.03,
                    #subplot_titles=('Subplot title1',  'Subplot title2', 'title3', 'title 4')
                    )
#### energy #####
# original data points
if on_orig:
    fig.append_trace(
        go.Scatter(
            mode='markers',
            x=get_original_datapoints('energy')['energy'], 
            y=get_original_datapoints('energy')['cutofenergy'], 
            name="energy dataset"), 
        row = 1, col=1)

to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "energy", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
#scaling law data points
y_pred_scale = get_test_scaling_law_pred_from_train(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "energy", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), X_train, y_train, n_mean = n_mean)
y_pred_scale['x'] = to_predict['energy']

fig.append_trace(
    go.Scatter(
        
        x=y_pred_scale['x'], 
        y=y_pred_scale['y_pred'], 
        name="energy",
        marker=dict(
            color='Gray',
        ),
    ), 
    row = 1, col=1)
# ml plot
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "energy", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'energy', slider_monotone_weight=slider_monotone_weight)
fig.append_trace(
    go.Scatter(
        
        x=df['x'], 
        y=smoother(df['y'], slider_smooth_power), 
        name="energy",
    ), 
    row = 1, col=1)

#### pulsewidth #####
# original data points
if on_orig:
    fig.append_trace(
        go.Scatter(
            mode='markers',
            x=get_original_datapoints('pulsewidth')['pulsewidth'], 
            y=get_original_datapoints('pulsewidth')['cutofenergy'], 
            name="pulsewidth dataset"), 
        row = 1, col=2)
# predicted model data points
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
#scaling law data points
y_pred_scale = get_test_scaling_law_pred_from_train(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), X_train, y_train, n_mean = n_mean)
y_pred_scale['x'] = to_predict['pulsewidth']
fig.append_trace(go.Scatter(x=y_pred_scale['x'], y=y_pred_scale['y_pred'], name="pulsewidth",marker=dict(
            color='Gray',
        ),), row = 1, col=2)
#ml plot
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'pulsewidth', slider_monotone_weight=slider_monotone_weight)
fig.append_trace(go.Scatter(x=df.x, y=smoother(df['y'], slider_smooth_power), name="pulsewidth"), row = 1, col=2)
#### spotsize #####
# original data points
if on_orig:
    fig.append_trace(
        go.Scatter(
            mode='markers',
            x=y_pred_scale['x'],
            y=y_pred_scale['y_pred'], 
            name="spotsize dataset"), 
        row = 2, col=1)
# predicted 
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
#scaling law data points
y_pred_scale = get_test_scaling_law_pred_from_train(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), X_train, y_train, n_mean = n_mean)
y_pred_scale['x'] = to_predict['spotsize']

fig.append_trace(go.Scatter(x=y_pred_scale['x'], y=y_pred_scale['y_pred'], name="spotsize", marker=dict(
            color='Gray',
        ),), row = 2, col=1)
# ml plot
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'spotsize', slider_monotone_weight=slider_monotone_weight)
fig.append_trace(go.Scatter(x=df.x, y=smoother(df['y'], slider_smooth_power), name="spotsize"), row = 2, col=1)

#### targetthickness #####
# original data points
if on_orig:
    fig.append_trace(
        go.Scatter(
            mode='markers',
            x=get_original_datapoints('targetthickness')['targetthickness'], 
            y=get_original_datapoints('targetthickness')['cutofenergy'], 
            name="targetthickness dataset"), 
        row = 2, col=2)
# predicted
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
#scaling law data points
y_pred_scale = get_test_scaling_law_pred_from_train(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), X_train, y_train, n_mean = n_mean)
y_pred_scale['x'] = to_predict['targetthickness']
fig.append_trace(go.Scatter(x=y_pred_scale.x, y=y_pred_scale.y_pred, name="targetthickness",marker=dict(
            color='Gray',
        ),), row = 2, col=2)

#ml plots
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'targetthickness', slider_monotone_weight=slider_monotone_weight)
fig.append_trace(go.Scatter(x=df.x, y=smoother(df['y'], slider_smooth_power), name="targetthickness"), row = 2, col=2)

st.subheader("Cut of energy plots")
fig.update_layout(
                #title="",
                #hovermode='x unified',
                legend=dict(
                    xanchor="center",
                    yanchor="top",
                    y=-0.1,
                    x=0
                ),
                height=700,
                yaxis_range=[-10,85],
                yaxis2_range=[-10,85],
                yaxis3_range=[-10,85],
                yaxis4_range=[-10,85]
            )
fig['layout']['xaxis']['title']='energy'
fig['layout']['xaxis2']['title']='pulsewidth'
fig['layout']['xaxis3']['title']='spotsize'
fig['layout']['xaxis4']['title']='targetthickness'
st.plotly_chart(fig, theme="streamlit", use_container_width=True)