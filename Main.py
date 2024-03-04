#%%
import os
import sys
import inspect

import streamlit as st
from streamlit_shap import st_shap
import pickle
import pandas as pd
import sys
sys.path.append('../train')
import config  
import features as train_features
from PredModel import PredModel
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
import matplotlib.pyplot as plt
from functions import *
from streamlit_super_slider import st_slider

#
st.set_page_config(
    layout="wide",
    page_title="Cutoffenergy Predictor from model",
    page_icon="🪟",
)
##### Sidebar Model Selection
with st.sidebar:
    expname = st.text_input('MLFlow experiment name', value=st.session_state.expname if 'expname' in st.session_state else '')
    runname = st.text_input('MLFlow run name', value=st.session_state.runname if 'runname' in st.session_state else '')
    st.write("***")
    col1, col2 = st.columns(2)
st.session_state.runname = runname
runname = st.session_state.runname

st.session_state.expname = expname
expname = st.session_state.expname
##### Display "Landing Page" if no model selected
if not runname or not expname:
    st.title('Enter experiment and model in sidebar')
###### If Model is selected
if runname:

    # get datafile for "original" data points and min-max-Values for prediction
    datafile = os.path.join(os.path.dirname(os.getcwd()), "data/dataset.xlsx")
    #initiate class
    pred = PredModel(datafile, expname, [0.15, 0.25, 0.75], debug = False)
    pred.loadData()
    pred.getRuns(
        filter_string="attributes.run_name = '" + runname + "'"#agreeable-horse-725'"#'redolent-mole-530'"
    )

    pred.getModels()
    # ToDo: Methode für RunID generieren....
    run_id = list(pred.getTrainDatasets())[0]

    ##### Set Sessions with Model and Data for Sub-Pages
    model_dict = pred.model_dict
    st.session_state.model_dict = model_dict

    # set features
    # set model
    for model in list(model_dict):
        features = pred.getTrainFeatures(model_dict[model]['run'])
        model = model

    st.session_state.model = model
    
    loaded_model = model_dict[model]['loaded_model']
    st.session_state.loaded_model = loaded_model
    
    logged_model = model_dict[model]['logged_model']
    st.session_state.logged_model = logged_model

    mlalgorithm = model_dict[model]['run']['params.Algorithm']
    st.session_state.mlalgorithm = mlalgorithm

    mlframework = model_dict[model]['run']['params.Framework']
    st.session_state.mlframework = mlframework

    run = mlflow.get_run(model_dict[model]['run']['run_id'])
    artifact_uri = run.info.artifact_uri
    client = mlflow.MlflowClient()
    ###### Sidebar with sliders ######
    with st.sidebar:
        # Smoother
        slider_smooth_power = st.slider("smoother power",min_value=1, max_value=200)
        slider_smooth_weight = st.slider("smoother weight",min_value=0, max_value=100)

        # Control Monotonicity Weight if Stacking_Estimator "stacking-monotone" is algorithm in mlflow run
        if "stacking-monotone" in mlalgorithm:
            slider_monotone_weight = st.slider("monotonicity weight",min_value=0, max_value=100)
        else: 
            slider_monotone_weight = 0

    ###### Get Train-/Test DAta and store in Sessions
    ###
    # reset index, da bei einigen csv read doppelte indexe vorkommen (wieso auch immer)
    # drop=True, da sonst alter index als neue spalte gespeichert wird
    # für debug drop=False und indexe vergleichen über st.write df in col 1 und col2 spalten unten
    X_test = pd.read_csv(client.download_artifacts(run_id, "test_data/X_test.csv"), index_col=0).reset_index(drop=True)
    y_test = pd.read_csv(client.download_artifacts(run_id, "test_data/y_test.csv"), index_col=0).reset_index(drop=True)
    X_train = pd.read_csv(client.download_artifacts(run_id, "train_data/X_train.csv"), index_col=0).reset_index(drop=True)
    y_train = pd.read_csv(client.download_artifacts(run_id, "train_data/y_train.csv"), index_col=0).reset_index(drop=True)

    st.session_state['X_test'] = X_test

    st.session_state['y_test'] = y_test

    st.session_state['X_train'] = X_train

    st.session_state['y_train'] = y_train

    #y_pred = get_2d_predictions(mlalgorithm, data = X_test, model = loaded_model)

    ###### Display Single Prediction and Metrics in Sidebar ######
    with st.sidebar:
        if "stacking-monotone" in mlalgorithm:
            y_pred = get_2d_predictions(mlalgorithm, X_test, loaded_model, 'energy', test=True, slider_monotone_weight=slider_monotone_weight)
            
            col2.metric('Current RMSE', format(mean_squared_error(y_test, y_pred, squared=False), '.3f'))
            col1.metric('Model RMSE', format(model_dict[model]['run']['metrics.RMSE'], '.3f'))
        else:
            col1.metric('Model RMSE', format(model_dict[model]['run']['metrics.RMSE'], '.3f'))
            y_pred = get_2d_predictions(mlalgorithm, train_features.X_test_raw, loaded_model, 'energy', test=True)
            col2.metric('RawTestSet RMSE', format(mean_squared_error(train_features.y_test_raw, y_pred, squared=False), '.3f'))
            y_pred = get_2d_predictions(mlalgorithm, train_features.X_test_wo_outliers, loaded_model, 'energy', test=True)
            st.metric('WoOutlierTestSet RMSE', format(mean_squared_error(train_features.y_test_wo_outliers, y_pred, squared=False), '.3f'))
    st.divider()
    ####### Parameter Selector #####
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

    with col1:
        st.markdown(':blue[Energy (J)]')
        x_energy = st_slider(min_value=float(config.data.min(numeric_only=True)['energy'].item()), max_value=max_energy)
        st.markdown(':green[Spot size (µm)]')
        x_spotsize = st_slider(min_value=float(config.data.min(numeric_only=True)['spotsize']) , max_value=max_spotsize)#float(config.data.quantile(q=0.99, axis=0, numeric_only=True)['spotsize']))
        on_orig = st.toggle('Show dataset')
    with col2:
        st.markdown(':red[Pulse width (fs)]')
        x_pulsewidth = st_slider(float(config.data.min(numeric_only=True)['pulsewidth']) ,max_pulsewidth)
        st.markdown(':violet[Target thickness (nm)]')
        x_targetthickness = st_slider(float(train_features.data.min(numeric_only=True)['targetthickness']) ,max_targetthickness)
    # calculate single prediction for display in sidebar
    # debug only
    x_energy = 2.4
    x_spotsize = 3.3
    x_pulsewidth = 31.3
    x_targetthickness = 635
    # x_energy = 3
    # x_spotsize = 5.3
    # x_pulsewidth = 500
    # x_targetthickness = 537.5

    df_single_pred = pd.DataFrame({ 
                        'energy' : [x_energy], 
                        'pulsewidth' : [x_pulsewidth],
                        'spotsize' : [x_spotsize],
                        'targetthickness' : [x_targetthickness],
                        })
    
    y = get_prediction(df_single_pred, mlalgorithm, loaded_model, slider_monotone_weight)
    if (type(y) == pd.DataFrame):
        single_prediction = y.values[0]
    elif  (type(y)) == np.ndarray:
        single_prediction = y[0]
    st.divider()
    st.markdown(":sparkles:**Predicted Cutoff-Energy**: "+ str(single_prediction))
    st.divider()
    #%%
    import numpy as np

    # y_pred = loaded_regressor(pd.DataFrame([{'spotsize':x_spotsize, 'energy':x_energy, 'pulsewidth':x_pulsewidth, 'targetthickness':x_targetthickness}]))

    fig = make_subplots(rows = 2, cols=2, start_cell="top-left", horizontal_spacing = 0.03
                        #subplot_titles=('Subplot title1',  'Subplot title2', 'title3', 'title 4')
                        )
    #### energy #####
    # original data points
    if on_orig:
        orig_df = get_original_datapoints('energy', data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
        orig_df = orig_df.loc[orig_df['energy'] < max_energy]
        fig.append_trace(
            go.Scatter(
                mode='markers',
                x=orig_df['energy'], 
                y=orig_df['cutoffenergy'], 
                marker=dict(color='rgba(0, 0, 0, 0.3)',),
                name="energy dataset"), 
            row = 1, col=1)
    # predicted model data points
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
        orig_df = get_original_datapoints('pulsewidth', data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
        orig_df = orig_df.loc[orig_df['pulsewidth'] < max_pulsewidth]
        fig.append_trace(
            go.Scatter(
                mode='markers',
                x=orig_df['pulsewidth'], 
                y=orig_df['cutoffenergy'], 
                marker=dict(color='rgba(0, 0, 0, 0.3)',),
                name="pulsewidth dataset"), 
            row = 1, col=2)
    # predicted model data points
    df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "pulsewidth", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'pulsewidth', slider_monotone_weight=slider_monotone_weight)
    fig.append_trace(go.Scatter(x=df.x, y=smoother(df['y'], slider_smooth_power), name="pulsewidth"), row = 1, col=2)


    #### spotsize #####
    # original data points
    if on_orig:
        orig_df = get_original_datapoints('spotsize', data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
        orig_df = orig_df.loc[orig_df['spotsize'] < max_spotsize]
        fig.append_trace(
            go.Scatter(
                mode='markers',
                x=orig_df['spotsize'], 
                y=orig_df['cutoffenergy'], 
                marker=dict(color='rgba(0, 0, 0, 0.3)',),
                name="spotsize dataset"), 
            row = 2, col=1)
    # predicted 
    df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "spotsize", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'spotsize', slider_monotone_weight=slider_monotone_weight)
    fig.append_trace(go.Scatter(x=df.x, y=smoother(df['y'], slider_smooth_power), name="spotsize"), row = 2, col=1)

    #### targetthickness #####
    # original data points
    if on_orig:
        orig_df = get_original_datapoints('targetthickness', data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, max_energy, max_pulsewidth, max_spotsize, max_targetthickness)
        orig_df = orig_df.loc[orig_df['targetthickness'] < max_targetthickness]
        fig.append_trace(
            go.Scatter(
                mode='markers',
                x=orig_df['targetthickness'], 
                y=orig_df['cutoffenergy'], 
                marker=dict(color='rgba(0, 0, 0, 0.3)',),
                name="targetthickness dataset"), 
            row = 2, col=2)
    # predicted
    df = get_2d_predictions(mlalgorithm, get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, "targetthickness", max_energy, max_pulsewidth, max_spotsize, max_targetthickness), loaded_model, 'targetthickness', slider_monotone_weight=slider_monotone_weight)
    fig.append_trace(go.Scatter(x=df.x, y=smoother(df['y'], slider_smooth_power), name="targetthickness"), row = 2, col=2)


    st.subheader("Cutoff-Energy plots")
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
    fig['layout']['xaxis']['title']='energy (J)'
    fig['layout']['xaxis2']['title']='pulsewidth (fs)'
    fig['layout']['xaxis3']['title']='spotsize (µm)'
    fig['layout']['xaxis4']['title']='targetthickness (nm)'
    # fig['layout']['yaxis']['title']='cutoffenergy'
    # fig['layout']['yaxis2']['title']='cutoffenergy'

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


##### residual plot #####
    st.subheader("Residual plot")
    if "TFLattice" in mlalgorithm:
        y_pred = loaded_model.predict([X_test['energy'], X_test['pulsewidth'], X_test['spotsize'], X_test['targetthickness']])
    elif "stacking-monotone" in mlalgorithm:
        y_pred = loaded_model.predict(get_X_with_weights(X_test, slider_monotone_weight))
    else:
        y_pred = loaded_model.predict(X_test)
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, columns=['predict'])
    if not 'predict' in y_pred.columns:
        y_pred = y_pred.rename(columns={0: 'predict'})
    residuals = pd.DataFrame()
    residuals['residual'] = y_test['cutoffenergy'] - y_pred['predict']

    fig_residuals_cutoff =go.Figure(data=go.Scatter(
                    mode='markers',
                    x=y_pred['predict'], 
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
    # ToDo: residual plot für alle features (fraglich ob Sinnvoll)
    # fig_residuals_energy =go.Figure(data=go.Scatter(
    #                 mode='markers',
    #                 x=y_pred['predict'], 
    #                 y=residuals['residual'], 
    #                 marker=dict(
    #                     color='rgba(99, 110, 250, 0.3)',
    #                     size=8,
    #                 ),
    #                 name="energy dataset"))
    # fig_residuals_energy.update_layout(
    #     title="Residuals vs Cutoffenergy",
    #     xaxis_title="Cutoffenergy (MeV)",
    #     yaxis_title="Residual",
    #     #legend_title="Legend Title",
    # )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_residuals_cutoff, theme="streamlit", use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(residuals, x='residual', title="Histogram of Residuals", labels={'residual': 'Residuals'}), theme="streamlit", use_container_width=True)

##### Load Original MOdel #####
    if ('h2o' in str(loaded_model)):
        orig_model = mlflow.h2o.load_model(logged_model)
    if ('sklearn' in str(loaded_model)):
        orig_model = mlflow.sklearn.load_model(logged_model)
    if ('tensorflow' in str(loaded_model)):
        import mlflow.keras
        orig_model = mlflow.keras.load_model(logged_model)
    if ('pyfunc' in str(loaded_model) and 'TFLattice' in mlalgorithm):
        import mlflow.keras
        orig_model = mlflow.pyfunc.load_model(logged_model)

##### Feature Importance #####
    if ('sklearn' in str(loaded_model) and not 'stacking-monotone' in mlalgorithm):

        # check if model already is the estimator itself
        if 'XGBRegressor' in str(type(orig_model)):
            estimator = orig_model
        elif 'LGBMRegressor' in str(type(orig_model)):
            estimator = orig_model
        else:
            estimator = orig_model.model.estimator
        
        st.subheader("Tree Feature Importance")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            #check if orig_model has attribute feature_name_
            #if not, use feature_names_in_ instead

            
            #
            if hasattr(orig_model, 'feature_name_'):
                ax.barh(orig_model.feature_name_, orig_model.feature_importances_)
            else: 
                ax.barh(orig_model.feature_names_in_, orig_model.feature_importances_)
            # if 'LGMBRegressor' in str(type(orig_model)):
            #     ax.barh(orig_model.feature_name_, orig_model.feature_importances_)
            # else:
            #     ax.barh(orig_model.feature_name_, orig_model.feature_importances_)
                #ax.barh(orig_model.feature_names_in_, orig_model.feature_importances_)
            st.pyplot(fig)
        with col2:
            from sklearn.inspection import permutation_importance
            result = permutation_importance (estimator, X_train, y_train, n_repeats=10,random_state=228, n_jobs = 2)
            sorted_importances_idx = result.importances_mean.argsort()
            importances = pd.DataFrame(
                result.importances[sorted_importances_idx].T,
                columns=X_test.columns[sorted_importances_idx],
            )
            ax = importances.plot.box(vert=False, whis=10)
            ax.set_title("Permutation Importances (test set)")
            ax.axvline(x=0, color="k", linestyle="--")
            ax.set_xlabel("Decrease in accuracy score")
            ax.figure.tight_layout()
            st.pyplot(ax.figure)
        
        col1, col2 = st.columns(2)
        import lightgbm as lgb
        with col1:
            plt.rcParams["figure.figsize"] = (7,3)
            if (hasattr(orig_model, 'model')):
                if ('xgboost' in str.lower(str(type(orig_model.model)))):
                    import xgboost as xgb
                    fig = xgb.plot_importance(estimator, importance_type="gain", title="XGBoost Feature Importance (Gain)").figure
                    plt.grid(False)
                    st.pyplot(fig) 
                if ('lgbm' in str.lower(str(type(orig_model.model)))):
                    fig = lgb.plot_importance(estimator, importance_type="gain", figsize=(7,3), title="LightGBM Feature Importance (Gain)").figure
                    plt.grid(False)
                    st.pyplot(fig)
            
            elif ('xgboost' in str.lower(str(type(orig_model)))): 
                import xgboost as xgb
                fig = xgb.plot_importance(estimator, importance_type="gain", title="XGBoost Feature Importance (Gain)").figure
                plt.grid(False)
                st.pyplot(fig)
            elif ('lgbm' in str.lower(str(type(orig_model)))): 
                import lightgbm as lgb
                fig = lgb.plot_importance(estimator, importance_type="gain", figsize=(7,3), title="LightGBM Feature Importance (Gain)").figure
                plt.grid(False)
                st.pyplot(fig)
            
        with col2:
            plt.rcParams["figure.figsize"] = (7,3)
            if (hasattr(orig_model, 'model')):
                if ('xgboost' in str.lower(str(type(orig_model.model)))):
                    import xgboost as xgb
                    fig = xgb.plot_importance(orig_model.model.estimator, importance_type="weight", title="XGBoost Feature Importance (Split / Weight)").figure
                    plt.grid(False)
                    st.pyplot(fig)
                if ('lgbm' in str.lower(str(type(orig_model.model)))): 
                    import lightgbm as lgb
                    fig = lgb.plot_importance(orig_model.model.estimator, importance_type="split", figsize=(7,3), title="LightGBM Feature Importance (Split)").figure
                    plt.grid(False)
                    st.pyplot(fig)
            elif ('xgboost' in str.lower(str(type(orig_model)))):  
                plt.rcParams["figure.figsize"] = (7,3)
                fig = xgb.plot_importance(estimator, importance_type="weight", title="XGBoost Feature Importance (Split / Weight)").figure
                plt.grid(False)
                st.pyplot(fig)
            elif ('lgbm' in str.lower(str(type(orig_model)))):
                import lightgbm as lgb
                fig = lgb.plot_importance(estimator, importance_type="split", figsize=(7,3), title="LightGBM Feature Importance (Split)").figure
                plt.grid(False)
                st.pyplot(fig)
        
    #plt.barh(automl.feature_names_in_, automl.feature_importances_)
    # st.write(loaded_model.predict(get_data_to_predict("targetthickness")))
    # st.write(get_data_to_predict('targetthickness'))

    #st.write(get_original_datapoints('energy'))
    # import pygwalker as pyg
    # pyg_html = pyg.walk(config.data, return_html=True)

    # # Embed the HTML into the Streamlit app
    # st.components.v1.html(pyg_html, height=1000, scrolling=True)

    ###### SHAP #####
    fig = plt.figure()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    # Create a SHAP explainer object
    if 'sklearn' in str.lower(mlframework) and not 'stacking-monotone' in mlalgorithm:
        explainer = shap.TreeExplainer(orig_model, X_train)
        # Compute SHAP values
        shap_values = explainer(X_train)
        shap_values_single = explainer.shap_values(df_single_pred)
    if 'stacking-monotone' in mlalgorithm:
        
        X_test_feature_names = pd.DataFrame(get_X_with_weights(X_test, slider_monotone_weight), columns=['energy', 'targetthickness', 'pulsewidth', 'spotsize', 'mono_weight_1', 'mono_weight_2'])

        #get_stack_prediction = lambda x: loaded_model.predict(get_X_with_weights(x, slider_monotone_weight)) # not needed/ data with monotone_weiths is enouth
        explainer = shap.KernelExplainer(loaded_model.predict, X_test_feature_names) #x_test for faster results
        shap_values=explainer(X_test_feature_names)
        shap_values = shap_values[:, :4]
        X_single_pred_names = pd.DataFrame(get_X_with_weights(df_single_pred, slider_monotone_weight), columns=['energy', 'targetthickness', 'pulsewidth', 'spotsize', 'mono_weight_1', 'mono_weight_2'])
        #shap_values_single = explainer.shap_values(X_single_pred_names)[:, :4]
        shap_values_single = explainer.shap_values(X_single_pred_names)


        #shap_values_single = shap_values_single[:, :4]
    if 'flaml' in str.lower(mlframework):
        explainer = shap.KernelExplainer(loaded_model.predict, X_train)
        # Compute SHAP values
        shap_values = explainer(X_test)
        shap_values_single = explainer.shap_values(df_single_pred)
    if 'h2o' in str.lower(mlframework):
        import h2o
        test_df = pd.concat([X_test, y_test], axis=1)
        test_h2oframe = h2o.H2OFrame(test_df)
        # contributions = orig_model.predict_contributions(test_h2oframe)
        # # convert the H2O Frame to use with shap's visualization functions
        # contributions_matrix = contributions.as_data_frame().values()
        # shap values are calculated for all features
        # shap_values = contributions_matrix[:,0:13]
        # # expected values is the last returned column
        # expected_value = contributions_matrix[:,13].min()
        # Assuming `model` is your trained H2O model and `frame` is the H2O frame you want to get the Shapley values for
        contributions = orig_model.predict_contributions(test_h2oframe)

        # Convert the H2O Frame to a pandas DataFrame
        contributions_df = contributions.as_data_frame()

        # The column names of the contributions are the feature names
        feature_names = contributions_df.columns.tolist()[:-1]  # Exclude the last column ('BiasTerm')

        # Get the Shapley values
        shap_values = contributions_df[feature_names].values
        
    if 'flaml' in str.lower(mlframework):
        
        explainer = shap.TreeExplainer(orig_model.model.estimator, X_train)
        
    if 'tensorflow' in str.lower(mlframework) and 'TFLattice' not in mlalgorithm:
        explainer = shap.KernelExplainer(orig_model.predict, X_train)
    if 'TFLattice' in mlalgorithm:

        orig_model = mlflow.tensorflow.load_model(logged_model)
        import tensorflow as tf

        def new_predict_fn(x):
            # Split the input tensor into 4 separate tensors
            inputs = tf.split(x, num_or_size_splits=4, axis=1)
            return loaded_model.predict(inputs)
        
        #explainer = shap.KernelExplainer(new_predict_fn, data_df)
        # Summarize the background data

        #sample for faster shap values and explainer
        #background_data = shap.sample(X_test, 10)
        background_data = X_test

        #bd_df = pd.concat([background_data['energy'], background_data['pulsewidth'], background_data['spotsize'], background_data['targetthickness']], axis=1)
        #background_data = [bd_df[name].values for name in ['energy', 'pulsewidth', 'spotsize', 'targetthickness']]

        # Create the SHAP explainer
        explainer = shap.KernelExplainer(new_predict_fn, background_data)
        
        # # Convert each Pandas Series to a NumPy array and reshape it
        # energy = X_test['energy'].values#.reshape(-1, 1)
        # pulsewidth = X_test['pulsewidth'].values#.reshape(-1, 1)
        # spotsize = X_test['spotsize'].values#.reshape(-1, 1)
        # targetthickness = X_test['targetthickness'].values#.reshape(-1, 1)
        # st.write(spotsize)
        # # Create the SHAP explainer // deep explainer not compatible with TFLattice
        # explainer = shap.KernelExplainer(new_predict_fn, background_data)
        shap_values = explainer.shap_values(background_data)
        shap_object = explainer(background_data)
        #explanation = shap.Explanation(shap_values, data=background_data)

        #base_values = explainer.expected_value(background_data)
        st.write(background_data)  
        st.write(new_predict_fn(background_data))
    # Compute SHAP values
        
    # if not shap_values:
    #     shap_values = explainer(X_test)
        #st.write(shap_values)
    # Stunden meines Lebens vergeudet bis ich diesen GitHub post gefunden habe: https://github.com/shap/shap/issues/1460#issuecomment-889292713
    if 'TFLattice' in mlalgorithm:
        #### tensorflow lattice ---
        st_shap(shap.violin_plot(shap_values=np.take(shap_object.values,0,axis=-1), features = background_data, feature_names = background_data.columns, sort = False))
        #st_shap(shap.plots.beeswarm(shap_values[:,:,1]))
    else:
        st_shap(shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=10))
        clear_fig()
        st_shap(shap.plots.bar(shap_values))
        #st_shap(shap.force_plot(shap_values, X_test), 500)
        clear_fig()
        # Create a summary plot
        col1, col2 = st.columns(2)
        with col1:
            st_shap(shap.plots.violin(shap_values))
            clear_fig()
        with col2:
            st_shap(shap.plots.beeswarm(shap_values))
            clear_fig()
        
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.markdown("#### SHAP heatmap (sorted by prediction)")
        #     y_pred = get_2d_predictions(mlalgorithm, train_features.X_test_raw, loaded_model, 'energy', test=True)
        #     order = np.argsort(y_pred['y'].values)
        #     st_shap(shap.plots.heatmap(shap_values, instance_order=order))
        #     clear_fig()
        # with col2:
        #     st.markdown("#### SHAP heatmap (sorted by energy)")
        #     order = np.argsort(X_test['energy'].values)
        #     st_shap(shap.plots.heatmap(shap_values, instance_order=order))
        #     clear_fig()
        #     st_shap(shap.plots.heatmap(shap_values))
        # st_shap(shap.force_plot(explainer.expected_value, shap_values_single, df_single_pred)) 
        fig = plt.figure()
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        #### waterfall plot workaround...
        shap_values_single = explainer(df_single_pred)
        shap_object = shap.Explanation(base_values = shap_values_single[0][0].base_values,
        values = shap_values_single[0].values,
        feature_names = X_test.columns,
        data = shap_values_single[0].data)
        st_shap(shap.plots.waterfall(shap_object))
        ####
        st_shap(shap.plots.scatter(shap_values[:, "energy"], color=shap_values[:, 'targetthickness']))#color=shap_values[:, 'spotsize']
        clear_fig()
        st_shap(shap.plots.scatter(shap_values[:, "pulsewidth"], color=shap_values))#color=shap_values[:, 'spotsize']
        clear_fig()
        st_shap(shap.plots.scatter(shap_values[:, "spotsize"], color=shap_values))#color=shap_values[:, 'spotsize']
        clear_fig()
        st_shap(shap.plots.scatter(shap_values[:, "targetthickness"], color=shap_values))#color=shap_values[:, 'spotsize']
        clear_fig()
        
        #st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values_single[0]))
    # st_shap(shap.dependence_plot("energy", shap_values.values, X_test))
    ### tensorflow lattice
    if 'tfl' in str.lower(mlalgorithm):
        st_shap(shap.dependence_plot("pulsewidth", shap_values[0], background_data))
        st_shap(shap.dependence_plot("spotsize", shap_values.values, X_test))
        st_shap(shap.dependence_plot("targetthickness", shap_values.values, X_test))
    #@knoam You can use sklearn.inspection.permutation_importance on automl.model.estimator or automl.best_model_for_estimator['lgbm'].estimator

    #%%
    # Fits the explainer
    # shap_values = shap.KernelExplainer(loaded_model.predict, X_test).shap_values(X_test)
    # #shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test)
    # Calculates the SHAP values - It takes some time
    #st.write(type(mlflow.sklearn.load_model(logged_model)))

    
    #st.write((find_closest_rows(X_test, 100, 900, 100, 1).name))
# st.write(y_test[y_test.index.duplicated(keep=False)])
# y_pred_scale = get_test_scaling_law_pred_from_train(X_test, X_train, y_train, n_mean = 9)
# st.plotly_chart(px.scatter(pd.concat([y_test, y_pred_scale], axis=1), x='cutoffenergy', y='y_pred'), theme="streamlit", use_container_width=True)



# st.write("RMSE: " + str(mean_squared_error(y_test, y_pred_scale, squared=False)))
    

### ab hier alles auskommentiert
#     h2omod = mlflow.h2o.load_model(logged_model) 
#     test = pd.concat([X_test, y_test], axis=1)
#     #st.write(test)
#     import h2o
#     test = h2o.H2OFrame(test) 
   
#     st.write(type(h2omod.explain(test)))

#     explainer = shap.TreeExplainer(mlflow.sklearn.load_model(logged_model).named_steps['lgbm'])
#     shap_values = explainer.shap_values(mlflow.sklearn.load_model(logged_model)[:-1].transform(X_train))
# #https://github.com/shap/shap/issues/1373#issuecomment-787999991

#     # shap_values = explainer.shap_values(X_train)
#     #loaded_model
#     # loaded_model.unwrap_python_model()
#     # explainer = shap.Explainer(loaded_model.predict, X_train)
#     # shap_values = explainer(X_train)
#     #st_shap(shap.force_plot(explainer.expected_value, shap_values[900,:], X_train.iloc[900,:]))

#    # st_shap(shap.force_plot(explainer.expected_value,  shap_values[1020], X_train.iloc[1020]))
#     #st_shap(shap.force_plot(explainer.expected_value, shap_values[:1444,:], X_train.iloc[:1444,:]), height=1000, width=1920)
#     st_shap(shap.force_plot(explainer.expected_value, shap_values, X_train), height=900, width=900)
#     #st_shap(shap.summary_plot(explainer.expected_value, shap_values[:500,:], X_train.iloc[:500,:]), height=900, width=500)
#     #shap.force_plot(explainer.expected_value, shap_test.values, X_test)
#     #st.write('test')
#     #st_shap(shap.summary_plot(shap_values))
#     #%%
#     ex = shap.KernelExplainer(loaded_model.predict, X_train)
#     shap_values = ex.shap_values(X_test)
#     st_shap(shap.summary_plot(shap_values, X_test))
# #https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
#     #%% 
#     import seaborn as sns
#     ### Plotting helper functions

#     # def plot_residual_distribution(model, x, y_pred, y_actual):
#     #     """
#     #     Density plot of residuals (y_true - y_pred) for testation set for given model 
#     #     """
#     #     ax = sns.kdeplot(y_test - model.predict(X_test))
#     #     title = ax.set_title('Kernel density of residuals')

#     # def plot_scatter_pred_actual(model, x, y_actual):
#     #     """
#     #     Scatter plot of predictions from given model vs true target variable from testation set
#     #     """
#     #     if "tensorflow" in mlalgorithm:
#     #         x_pred= model.predict((x['energy'], x['pulsewidth'], x['spotsize'], x['targetthickness']))[:,0] #[:,0] für 1. spalte aus mehrdimensionalem array
#     #     else: 
#     #         x_pred = model.predict(x)
#     #     ax = sns.scatterplot(x=x_pred, y = y_actual)
#     #     ax.set_xlabel('Predictions')
#     #     ax.set_ylabel('Actuals')
#     #     title = ax.set_title('Actual vs Prediction scatter plot')  
#     # plot_scatter_pred_actual(loaded_model, config.data[['energy', 'pulsewidth', 'spotsize', 'targetthickness']], config.data['cutoffenergy'])


#     # col1, col2 = st.columns(2)

#     # with col1:
#     #     df = get_2d_predictions(mlalgorithm, get_data_to_predict("energy"), loaded_model, 'energy')
#     #     px_plot = px.scatter(get_original_datapoints('energy'),x=['energy'],
#     #                         y='cutoffenergy',
#     #                         marginal_x="histogram")
#     #     px_plot.add_scatter(x=df.x, y=df.y)
#     #     px_plot.update_layout(
#     #                 yaxis_range=[-5,60],
#     #                 showlegend=False
#     #     )
#     #     st.plotly_chart(px_plot, theme="streamlit", use_container_width=True)

#     #     df = get_2d_predictions(mlalgorithm, get_data_to_predict("spotsize"), loaded_model, 'spotsize')
#     #     px_plot = px.line(df, x='x', y='y')
#     #     px_plot.update_layout(
#     #                 yaxis_range=[-5,60]
#     #     )
#     #     st.plotly_chart(px_plot, theme="streamlit", use_container_width=True)
# %%
