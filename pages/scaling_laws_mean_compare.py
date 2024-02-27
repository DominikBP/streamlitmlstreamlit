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
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test

st.header("Average Scaling Law Cutoffenergy Comparison")

if not os.path.exists('./temp/metric_df.csv'):
    rmse = dict()
    r2 = dict()
    mae = dict()

    for n_mean in range(1, 41):
        y_pred_scale = get_test_scaling_law_pred_from_train(X_test, X_train, y_train, n_mean)
        rmse[n_mean] = mean_squared_error(y_test, y_pred_scale, squared=False)
        r2[n_mean] = r2_score(y_test, y_pred_scale)
        mae[n_mean] = mean_absolute_error(y_test, y_pred_scale)

    metric_df = pd.DataFrame({'RMSE': rmse, 'R2': r2, 'MAE': mae})
    
    st.write(metric_df)
    metric_df.to_csv('./temp/metric_df.csv')

else:
    metric_df = pd.read_csv('./temp/metric_df.csv', index_col=0)
    #st.write(metric_df)

    metric_df = metric_df.round(3)
    fig = px.bar(
                metric_df, x=metric_df.index, y=metric_df.columns,barmode='group',text_auto=True,
                labels={
                    'index':"n",
                },
                title="Scaling Law Cutoffenergy prediction n_mean vs Metrics"
            )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    temp = metric_df
    temp['1-R2'] = 1-temp['R2']
    temp = temp.drop('R2', axis=1, inplace=False)
    st.write(temp)
    

    fig = px.bar(
                temp, x=temp.index, y=temp.columns,
                labels={
                    'index':"n",
                },
                title="Scaling Law Cutoffenergy prediction n_mean vs Metrics"
            )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    

    fig = px.bar(
            metric_df, x=metric_df.index, y=metric_df.RMSE,text_auto=True,
            labels={
                'index':"n",
            },
            title="Scaling Law Cutoffenergy prediction n_mean vs RMSE"
            
        )
    #fig.add_hline(y=metric_df.RMSE.min(), line_dash="dot",annotation_text="min RMSE", annotation_position="top right")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    fig = px.bar(
                metric_df, x=metric_df.index, y=metric_df.R2,
                labels={
                    'index':"n",
                },
                title="Scaling Law Cutoffenergy prediction n_mean vs R2"
            )
        
    fig.add_hline(y=metric_df.R2.max(), line_dash="dot", annotation_text="max R2", annotation_position="top right")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    fig = px.bar(
                metric_df, x=metric_df.index, y=metric_df.MAE,
                labels={
                    'index':"n",
                },
                title="Scaling Law Cutoffenergy prediction n_mean vs MAE"
            )
        
    fig.add_hline(y=metric_df.MAE.min(), line_dash="dot", annotation_text="min MAE", annotation_position="top right")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    # st.plotly_chart(px.scatter(pd.concat([y_test, y_pred_scale], axis=1), x='Cutoffenergy', y='y_pred'), theme="streamlit", use_container_width=True)
    
    # n mean = 5 residual plot scaling law
    y_pred_scale = get_test_scaling_law_pred_from_train(X_test, X_train, y_train, 5)
    ##### residual plot #####
    st.subheader("Residual plot")
    if not isinstance(y_pred_scale, pd.DataFrame):
        y_pred_scale = pd.DataFrame(y_test, y_pred_scale, columns=['predict'])
    if not 'predict' in y_pred_scale.columns:
        y_pred_scale = y_pred_scale.rename(columns={0: 'predict'})
    residuals = pd.DataFrame()
    y_pred_scale['predict'] = y_pred_scale['y_pred']
    residuals['residual'] = y_test['cutoffenergy'] - y_pred_scale['predict']

    fig_residuals_cutoff =go.Figure(data=go.Scatter(
                    mode='markers',
                    x=y_pred_scale['predict'], 
                    y=residuals['residual'], 
                    marker=dict(
                        color='rgba(99, 110, 250, 0.3)',
                        size=8,
                    ),
                    name="energy dataset"))
    fig_residuals_cutoff.update_layout(
        title="Residuals vs Cutoff-Energy",
        xaxis_title="Cutoff-Energy (MeV)",
        yaxis_title="Residual",
        yaxis_range=[-25,25],
    )
   
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_residuals_cutoff, theme="streamlit", use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(residuals, x='residual', title="Histogram of Residuals", labels={'residual': 'Residuals'}), theme="streamlit", use_container_width=True)
