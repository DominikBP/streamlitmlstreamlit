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

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=False,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test

loaded_model = st.session_state.loaded_model
mlalgorithm = st.session_state.mlalgorithm

st.header("Average Scaling Law Cutofenergy Comparison")
st.subheader("Reference Point selector")
selection = dataframe_with_selections(pd.concat([X_train, y_train], axis=1))

st.subheader("Parameter selector") 

col1, col2 = st.columns(2)

data = config.data

max_energy = float(400) #140
max_pulsewidth = float(data.quantile(q=0.95, axis=0, numeric_only=True)['pulsewidth'])
max_spotsize = float(12)
max_targetthickness = float(data.quantile(q=0.9, axis=0, numeric_only=True)['targetthickness']) 
n_mean = 2
with col1:
    x_energy = st.slider("Energy in Joule",min_value=float(config.data.min(numeric_only=True)['energy'].item()), max_value=max_energy)
    x_spotsize = st.slider("Spot size", float(config.data.min(numeric_only=True)['spotsize']) , max_spotsize)#float(config.data.quantile(q=0.99, axis=0, numeric_only=True)['spotsize']))
    on_orig = st.toggle('display original data points')
with col2:
    x_pulsewidth = st.slider("Pulse width",float(config.data.min(numeric_only=True)['pulsewidth']) ,max_pulsewidth)
    x_targetthickness = st.slider("Target thickness in mq",0.01 ,max_targetthickness)

    df_single_pred = pd.DataFrame({ 
                        'energy' : [x_energy], 
                        'pulsewidth' : [x_pulsewidth],
                        'spotsize' : [x_spotsize],
                        'targetthickness' : [x_targetthickness],
                        })
    
    # y = get_prediction(df_single_pred)
    # st.write(str(y.values[0]))

with st.sidebar:
        
    slider_smooth_power = st.slider("smoother power",min_value=1, max_value=200)
    slider_smooth_weight = st.slider("smoother weight",min_value=0, max_value=100)
    if "stacking-monotone" in mlalgorithm:
        slider_monotone_weight = st.slider("monotonicity weight",min_value=0, max_value=100)
st.divider()

fig = make_subplots(rows = 2, cols=2, start_cell="top-left",
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
# scaling law data points
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "energy", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
y_pred_scale = get_test_scaling_law_pred_from_reference_point(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "energy", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), selection)
y_pred_scale['x'] = to_predict['energy']

fig.append_trace(
    go.Scatter(
        
        x=y_pred_scale['x'], 
        y=y_pred_scale['calculated_ec'], 
        name="energy",
        marker=dict(
            color='Gray',
        ),
    ), 
    row = 1, col=1)
# ml data points
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "energy", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'energy')
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
# scaling law datapoints
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
y_pred_scale = get_test_scaling_law_pred_from_reference_point(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), selection)
y_pred_scale['x'] = to_predict['pulsewidth']
fig.append_trace(
    go.Scatter(
        x=y_pred_scale['x'], 
        y=y_pred_scale['calculated_ec'], 
        name="pulsewidth",
        marker=dict(
            color='Gray',
        ),
    ), 
    row = 1, col=2
)
# ml data points
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, "pulsewidth")
fig.append_trace(
    go.Scatter(
        
        x=df['x'], 
        y=smoother(df['y'], slider_smooth_power), 
        name="pulsewidth",
    ), 
    row = 1, col=2)
#### spotsize #####
# original data points
if on_orig:
    fig.append_trace(
        go.Scatter(
            mode='markers',
            x=y_pred_scale['x'],
            y=y_pred_scale['calculated_ec'], 
            name="spotsize dataset"), 
        row = 2, col=1)
# scaling law datapoints
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
y_pred_scale = get_test_scaling_law_pred_from_reference_point(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), selection)
y_pred_scale['x'] = to_predict['spotsize']

fig.append_trace(
    go.Scatter(
        x=y_pred_scale['x'], 
        y=y_pred_scale['calculated_ec'], 
        name="spotsize",
        marker=dict(
            color='Gray',
        ),
    ), 
    row = 2, col=1
)
# ml data points
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, "spotsize")
fig.append_trace(
    go.Scatter(
        
        x=df['x'], 
        y=smoother(df['y'], slider_smooth_power), 
        name="spotsize",
    ), 
    row = 2, col=1)
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
# scaling law datapoints
to_predict = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
y_pred_scale = get_test_scaling_law_pred_from_reference_point(get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), selection)
y_pred_scale['x'] = to_predict['targetthickness']
fig.append_trace(
    go.Scatter(
        x=y_pred_scale.x, 
        y=y_pred_scale.calculated_ec, 
        name="targetthickness",
        marker=dict(
            color='Gray',
        ),
    ), 
    row = 2, col=2
)
# ml data points
df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, "targetthickness")
fig.append_trace(
    go.Scatter(
        
        x=df['x'], 
        y=smoother(df['y'], slider_smooth_power), 
        name="targetthickness",
    ), 
    row = 2, col=2)
##### plot figure

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