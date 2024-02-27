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
# get train-/test data from session and selected model
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test

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

col1, col2 = st.columns(2)
with col1:
    st.subheader("Metrics on Test Data for selected Train Data Reference Point")
    selection = dataframe_with_selections(pd.concat([X_train, y_train], axis=1))
    st.write("Your selection:")
    st.write(selection)
    if not selection.empty:
        y_pred_scale= get_test_scaling_law_pred_from_reference_point(X_test, selection)
        st.write("RMSE: " + str(mean_squared_error(y_test, y_pred_scale, squared=False)))



with col2:
    st.subheader("RMSE on Test Data for different Train Data Reference Points")
    Xy_train = pd.concat([X_train, y_train], axis=1)
    Xy_train['scaling_test_rmse'] = Xy_train.apply(
                                        lambda row: 
                                            mean_squared_error(y_test, get_test_scaling_law_pred_from_reference_point(X_test, row), squared=False), 
                                            axis=1
                                    )
    Xy_train['scaling_test_mae'] = Xy_train.apply(
                                        lambda row: 
                                            mean_absolute_error(y_test, get_test_scaling_law_pred_from_reference_point(X_test, row)), 
                                            axis=1
                                    )
    Xy_train['scaling_test_r2'] = Xy_train.apply(
                                        lambda row: 
                                            r2_score(y_test, get_test_scaling_law_pred_from_reference_point(X_test, row)), 
                                            axis=1
                                    )
    fig = px.box(Xy_train, y='scaling_test_rmse')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.write (Xy_train)
    st.write(Xy_train.describe())
    # n_mean = st.slider("n mean",min_value=1, max_value=30)
    # y_pred_scale = get_test_scaling_law_pred_from_train(X_test, X_train, y_train, n_mean)
    # st.write("RMSE: " + str(mean_squared_error(y_test, y_pred_scale, squared=False)))
    # st.plotly_chart(px.scatter(pd.concat([y_test, y_pred_scale], axis=1), x='cutofenergy', y='y_pred'), theme="streamlit", use_container_width=True)
    
    