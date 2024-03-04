import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import mlflow
import os
import sys
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
class PredModel():
    """_summary_
    """
    def __init__(self, datafile="", exp="", quantiles=[0.25, 0.5, 0.75], debug=False):
        """_summary_

        Args:
            datafile (str, optional): _description_. Defaults to "".
            exp (str, optional): _description_. Defaults to "".
            quantiles (list, optional): _description_. Defaults to [0.25, 0.5, 0.75].
        """        
        self.debug = debug
        self.datafile = datafile
        self.exp = exp
        self.quantiles = quantiles

    def quickEval(self):
        self.loadData()
        self.getRuns()
        self.getModels()
        self.getPredictions()
        self.plot()

    def getTrainFeatures(self, run):
        features = list(run['params.Features'].replace("'", "").strip("][").split(", "))
        features.remove('cutofenergy') if 'cutofenergy' in features else ""
        return features

    def getRuns(self, filter_string="", exclude_feature=""):
        """_summary_
        """        
        #get runs
        mlflow.set_tracking_uri('http://localhost:5000')
        experiment_name = self.exp
        if self.debug == False:
            print("debug=false")
            runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)
            #print (runs)
        else:
            runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string, max_results=3)

        if exclude_feature:
            runs = runs[~runs["params.Features"].str.contains(exclude_feature, na=False)] #~inverts
        #print(runs)
        runs = runs[runs['metrics.RMSE'].notnull()]
        runs.sort_values(["metrics.RMSE"], ascending = [True], inplace = True)
        self.runs = runs

    def getModels(self):
        model_dict = collections.OrderedDict([])
        runs = self.runs
        # iterate each run and get model path / set artifact path in run array/list
        for index, run in runs.iterrows():
            artifact_uri = run['artifact_uri']

            # get list of dirs in artifact path of run
            listdir = os.listdir(artifact_uri)
            # check if either .models or automl in artifact path listdir
            # enter model "name" (foldername) into model column in runs
            #print (str(run['tags.mlflow.log-model.history']))
            #str_run = str(run['tags.mlflow.log-model.history']).strip("'<>() ").replace('\'', '\"')
            #print (type(str_run))
            str_run = str(run['tags.mlflow.log-model.history'])
            #tags =  json.loads(str(run['tags.mlflow.log-model.history'])) # cast to str sehr wichtig, sonst TypeError: the JSON object must be str, bytes or bytearray, not NoneType
                                    #tags.mlflow.log-model.history
            tags =  json.loads(str_run)
            artifact_path = tags[0]['artifact_path']
            runs.loc[int(index), "artifact_path"] = artifact_path

        for index, run in runs.iterrows():
            #print (run['artifact_uri'])
            logged_model = run['artifact_uri'] +"/"+run['artifact_path']
            print (logged_model)
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            model_dict[run.run_id] = {'loaded_model':loaded_model,'run':run, 'logged_model':logged_model}

        self.model_dict = model_dict

    def getTrainDatasets(self):
        # ToDo: Umstrukturieren bzw. getRunID Methode schreiben, da in der Main.py nur für RunID aufgerufen....
        
        dataset_dict = collections.OrderedDict([])
        runs = self.runs
        # iterate each run and get dataset.xlsx path / set artifact path in run array/list
        for index, run in runs.iterrows():
            training_excel_location = run['artifact_uri']+"/dataset/dataset.xlsx"
            ######## ToDo achtung hardcoded ###################
            # 
            #df = pd.read_excel(training_excel_location) 
            df = pd.read_excel('/home/dominik/Research-Incubator/train/artifacts/28/1a6a5c0bba0f48478ac9c9ed5a4cdf9e/artifacts/dataset/dataset.xlsx') 
            #### hardcoded
            df.drop(df.columns[0], axis=1, inplace=True) # lösche index spalte von erzeugter excel
            dataset_dict[run.run_id] = df

        self.dataset_dict = dataset_dict
        return dataset_dict

    def loadData(self):
        data = pd.read_excel(self.datafile)
        data.drop(data.columns[0], axis=1, inplace=True) # lösche index spalte von erzeugter excel
        self.data = data
        # dataframe collection quantiles
        data_collection = collections.OrderedDict([])
        quantiles = self.quantiles

        for quantile in quantiles:
            quantile_spotsize = float(data.quantile(q=quantile, axis=0, numeric_only=True)['spotsize'])
            quantile_energy = float(data.quantile(q=quantile, axis=0, numeric_only=True)['energy'])
            quantile_pulsewidth = float(data.quantile(q=quantile, axis=0, numeric_only=True)['pulsewidth'])
            quantile_targetthickness = float(data.quantile(q=quantile, axis=0, numeric_only=True)['targetthickness'])
            range_energy = np.arange(float(data.min()['energy']), float(data.max()['energy']) ,0.5)
            range_intensity = np.linspace(float(data.min()['intensity']), float(data.max()['intensity']) ,len(range_energy))
            data_collection[quantile] = pd.DataFrame({ 
                'intensity' : range_intensity,
                'energy' : range_energy,
                'spotsize' : [quantile_spotsize] * len (range_energy),
                'pulsewidth' : [quantile_pulsewidth] * len (range_energy),
                'targetthickness' : [quantile_targetthickness] * len (range_energy)
                })
        self.data_collection = data_collection

    def getPredictions(self):
        
        model_dict = self.model_dict
        for model in list(model_dict):
            data_collection = self.data_collection
            # todo: create predictions for each loaded model! maybe in separate dict/collection...
            loaded_model = model_dict[model]['loaded_model']
            features = self.getTrainFeatures(model_dict[model]['run'])
            # mlflow.pyfunc.load_model(model)
            # get predictions for dataframe collection
            # and collect predictions for quantiles in prediction_collection{}
            # print (model_dict[model]['run']['run_id'])
            prediction_collection = {}
            for quantile in list(data_collection): # iterate through dataframe_collection !!! no prediction_collection - as of now it's empty
                quan_data = data_collection[quantile]
                #only if params intensity!!!
                # if not "intensity" in features:
                #     if "intensity" in quan_data.columns:
                #         quan_data.drop(columns="intensity", inplace=True)
                # if not "energy" in features:
                #     if "energy" in quan_data.columns:
                #         quan_data.drop(columns="energy", inplace=True)
                # if "energy" in quan_data:
                #     quan_data.drop(columns="energy", inplace=True)
                print ("first"+str(quan_data.columns))
                if "cutofenergy" in quan_data.columns:
                    quan_data = quan_data.drop(columns="cutofenergy", inplace=False)
                print (features)
                if "intensity" not in features:
                    if "intensity" in quan_data.columns:
                        quan_data = quan_data.drop(columns="intensity", inplace=False)
                if "Laser" in features:
                    quan_data['Laser'] = "nan"
                else:
                    if "Laser" in quan_data.columns:
                        quan_data = quan_data.drop(columns="Laser", inplace=False)
                # reindex columns to match with model training
                columns_titles = features
                quan_data=quan_data.reindex(columns=columns_titles)
                prediction_collection[quantile] = loaded_model.predict(quan_data)
            model_dict[model]['predictions'] = prediction_collection
        
        self.model_dict = model_dict

    def plot(self, fix_y_axis=False):

        model_dict = self.model_dict
        data_collection = self.data_collection
        for model in list(model_dict):

            features = self.getTrainFeatures(model_dict[model]['run'])
            count = 0
            if "intensity" in features:
                count +=1
            if "energy" in features:
                count +=1
            if count >= 2:
                sys.exit(f"intensity and energy {features} {model_dict[model]['run']['run_id']}")

            x = "set later"

            plt.figure(figsize=(12,4))
            title = f"{model_dict[model]['run']['run_id']} - {model_dict[model]['run']['tags.mlflow.runName']} \\\n model = \\\n datasetcount = {model_dict[model]['run']['params.FeatureCount']}\\\n trainfeatures={str(features)}, \\\n RMSE = "+"{:.2f}".format(model_dict[model]['run']['metrics.RMSE'])
            plt.title(title)

            k = 0

            for quantile in list(model_dict[model]['predictions']):
                if "intensity" in features:
                    x = data_collection[quantile]['intensity']
                if k == 0:
                    if "energy" in features:
                        x = data_collection[quantile]['energy']
                        selected = []
                        i=0
                        data = self.data
                        for xval in x:
                            #df_closest = data.iloc[(data['points']-101).abs().argsort()[:1]]
                            search = data.loc[(data['energy'] >= xval-5) & (data['energy'] <= xval+5)]
                            closest = search.iloc[(search['energy']-xval).abs().argsort()[:1]]
                            selected.append(closest['cutofenergy'].item())
                            i+=1
                        # print((selected))
                        plt.scatter(x, selected, c="black", alpha=0.1)
                k += 1
                plt.scatter(x, model_dict[model]['predictions'][quantile], label="{:.2f} Quantil, Spotsize={:.2f}, Targetthickness={:.2f}, Pulsewidth={:.2f}".format(quantile, data_collection[quantile]['spotsize'][0], data_collection[quantile]['targetthickness'][0], data_collection[quantile]['pulsewidth'][0]))
                plt.xlabel("Intensity" if "Intensity" in features else "Energy")
                plt.ylabel("Cutofenergy")
                plt.ylim(0, 100)
            plt.legend()

    def plotly(self, fix_y_axis=False):

        model_dict = self.model_dict
        data_collection = self.data_collection

        for model in list(model_dict):

            # set figure using plotly graph objects as go
            fig = go.Figure()
            
            features = self.getTrainFeatures(model_dict[model]['run'])
            x = "set later"

            df = pd.DataFrame()
            selected = pd.DataFrame()
            k = 0
            for quantile in list(model_dict[model]['predictions']):
                k+=1
                if "intensity" in features:
                    x = data_collection[quantile]['intensity']
                if "energy" in features:
                    x = data_collection[quantile]['energy']
                    
                    i=0
                    data = self.data

                    if k == 1:
                        for xval in x:

                            #df_closest = data.iloc[(data['points']-101).abs().argsort()[:1]]
                            search = data.loc[(data['energy'] >= xval-5) & (data['energy'] <= xval+5)]
                            closest = search.iloc[(search['energy']-xval).abs().argsort()[:1]]
                            selected = pd.concat([selected, closest])#selected.append(closest)
                            i+=1
                            
                        fig.add_trace(
                            go.Scatter(
                                x=x, y=selected['cutofenergy'], 
                                mode='markers', customdata=selected,
                                name='Data',
                                # energy  pulsewidth  cutofenergy  spotsize  targetthickness     intensity Laser
                                hovertext=[selected.pulsewidth, selected.spotsize], 
                                hovertemplate="<br>Pulsewidth: %{customdata[1]}<br>"
                                            + "Spotsize: %{customdata[3]}<br>"
                                            + "Targetthickness: %{customdata[4]}",
                                marker=dict(
                                    size=selected.spotsize,
                                    color=selected.pulsewidth,
                                    
                                )
                            )
                    )
                    
                
                if isinstance(model_dict[model]['predictions'][quantile], pd.DataFrame):
                    df_pred_ce = model_dict[model]['predictions'][quantile]
                    df_pred_ce = df_pred_ce.rename(columns={"predict": "pred_cutofenergy"})
                else:
                    df_pred_ce = pd.DataFrame(model_dict[model]['predictions'][quantile], columns=["pred_cutofenergy"])
                
                x = pd.DataFrame(x)

                df = pd.concat([x,df_pred_ce], axis=1)

                tracename = "{:.2f} Quantil, Spotsize={:.2f}, Targetthickness={:.2f}, Pulsewidth={:.2f}".format(quantile, data_collection[quantile]['spotsize'][0], data_collection[quantile]['targetthickness'][0], data_collection[quantile]['pulsewidth'][0])

                fig.add_trace(go.Scatter(x=df.iloc[:,0], y=df.pred_cutofenergy, name=tracename, #x=df.energy
                                        hovertemplate="<br>"))

            if not 'params.best_model' in model_dict[model]['run']:
                modelname = "test"
            else:
                modelname = model_dict[model]['run']['params.best_model']
            title = f"{model_dict[model]['run']['run_id']} - {model_dict[model]['run']['tags.mlflow.runName']} \
                    model = {modelname} \
                    datasetcount = {model_dict[model]['run']['params.FeatureCount']}<br> \
                    trainfeatures={str(features)}, <br> \
                    RMSE = "+"{:.2f}".format(model_dict[model]['run']['metrics.RMSE'])

            fig.update_layout(
                title=title,
                hovermode='x unified',
                legend=dict(
                    xanchor="center",
                    yanchor="top",
                    y=-0.1,
                    x=0.5
                ),
                height=1000,
            )

            fig.show()

    def getFeatureImportance(self):
        
        model_dict = self.model_dict
        data_collection = self.data_collection

        for model in list(model_dict):

            # todo: create predictions for each loaded model! maybe in separate dict/collection...
            loaded_model = model_dict[model]['loaded_model']
            print(type(loaded_model._model_impl))
            # <class 'mlflow.h2o._H2OModelWrapper'>
            # <class 'flaml.automl.automl.AutoML'>
            fi = loaded_model._model_impl.h2o_model.varimp(use_pandas = True) # python_model.model.feature_importances_()

        self.model_dict = model_dict

    def plotFI(self):
        
        model_dict = self.model_dict
       
        for model in list(model_dict):

            # todo: create predictions for each loaded model! maybe in separate dict/collection...
            loaded_model = model_dict[model]['loaded_model']
            print(type(loaded_model._model_impl))
            # <class 'mlflow.h2o._H2OModelWrapper'>
            # <class 'flaml.automl.automl.AutoML'>
            if (str(type(loaded_model._model_impl)) == "<class 'mlflow.h2o._H2OModelWrapper'>"):
                loaded_model._model_impl.h2o_model.varimp_plot()
            if (str(type(loaded_model._model_impl)) ==  "<class 'flaml.automl.automl.AutoML'>"):
                automl = loaded_model._model_impl
                plt.barh(automl.feature_names_in_, automl.feature_importances_)
                plt.show()
            #print (fi)