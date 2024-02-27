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

from functions import *
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test

loaded_model = st.session_state.loaded_model

y_pred = loaded_model.predict(X_test)
st.write(y_test)
st.write(y_pred)
residuals = pd.DataFrame()
residuals['residual'] = y_test['cutoffenergy'] - y_pred['predict']
st.write(residuals)

fig =go.Figure(data=go.Scatter(
                mode='markers',
                
                x=y_pred['predict'], 
                y=residuals['residual'], 
                marker=dict(
                    color='rgba(99, 110, 250, 0.3)',
                    size=8,
                ),
                name="energy dataset"))

st.plotly_chart(fig, theme="streamlit")
# st.subheader("Residual Plot") 
# Xy_train = pd.concat([X_train, y_train], axis=1)
# Xy_train['scaling_test_rmse'] = Xy_train.apply(
#                                     lambda row: 
#                                         mean_squared_error(y_test, get_test_scaling_law_pred_from_reference_point(X_test, row), squared=False), 
#                                         axis=1
#                                 )

    