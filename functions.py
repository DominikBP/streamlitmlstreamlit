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
from sklearn.metrics import mean_squared_error

# intensität berechnen
#=energy*4*LN(2)/(pulsewidth*0.000000000000001*3.14*(spotsize*0.0001)^2)
#t for target, r for reference. Formula 7 from  Zimmer, M., Scheuren, S., Ebert, T., Schaumann, G., Schmitz, B., Hornung, J., Bagnoud, V., Rödel, C. & Roth, M. (2021). Analysis of laser-proton acceleration experiments for development of empirical scaling laws. Physical review. E, 104(4-2), 45210. https://doi.org/10.1103/PhysRevE.104.045210

def calc_cutofenergy_scaling_law_zimmer(r_cutofenergy, r_energy, t_energy, r_pulsewidth, t_pulsewidth, r_spotsize, t_spotsize, r_targetthickness, t_targetthickness):
    t_cutofenergy = r_cutofenergy * (t_energy/r_energy)**0.59 * (t_pulsewidth/r_pulsewidth)**-0.09 * (t_spotsize/r_spotsize)**-0.58 * (t_targetthickness/r_targetthickness)**-0.16 # pulsewidth in fs, spotsize in micrometer, targetthickness in micrometer
    return t_cutofenergy

from scipy.spatial import distance

def find_closest_rows(df, energy, pulsewidth, spotsize, targetthickness, n=1):
    input_params = np.array([energy, pulsewidth, spotsize, targetthickness])
    # eucledian distance to each rox
    df['distance'] = df.apply(lambda row: distance.euclidean(input_params, row.values), axis=1)

    #closest_row = df.loc[df['distance'].idxmin()]
    closest_rows = df.nsmallest(n, 'distance')
    df.drop(columns=['distance'], inplace=True)
    return closest_rows


def get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, x_axis = "energy", max_energy = 150, max_pulsewidth = 500, max_spotsize = 30, max_targetthickness = 600):
    #data = pred.data
    data_to_predict = pd.DataFrame()

    if x_axis == "energy":

        range_energy = np.arange(float(data.min(numeric_only=True)['energy']), max_energy , 1)

        pulsewidth = x_pulsewidth #* len (range_energy)
        spotsize = x_spotsize #* len (range_energy)
        targetthickness = x_targetthickness #* len(range_energy)

        data_to_predict = pd.DataFrame({ 
                        'energy' : range_energy,
                        'pulsewidth' : pulsewidth,
                        'spotsize' : spotsize,
                        'targetthickness' : targetthickness,

                        })
        
    if x_axis == "pulsewidth":
        
        pulsewidth = np.arange(float(data.min(numeric_only=True)['pulsewidth']), max_pulsewidth , 1)

        targetthickness = x_targetthickness #* len(pulsewidth)
        energy = x_energy #* len(pulsewidth)
        spotsize = x_spotsize #* len (pulsewidth)

        data_to_predict = pd.DataFrame({ 
                        'energy' : energy, 
                        'pulsewidth' : pulsewidth,
                        'spotsize' : spotsize,
                        'targetthickness' : targetthickness,
                        })
        
    if x_axis == "targetthickness":

        targetthickness = np.arange(float(data.min(numeric_only=True)['targetthickness']), max_targetthickness , 1)

        pulsewidth = x_pulsewidth #* len(targetthickness)
        energy = x_energy #* len(targetthickness)
        spotsize = x_spotsize #* len (targetthickness)

        data_to_predict = pd.DataFrame({ 
                        'energy' : energy, 
                        'pulsewidth' : pulsewidth,
                        'spotsize' : spotsize,
                        'targetthickness' : targetthickness,
                        }) 
    
    if x_axis == "spotsize":

        spotsize = np.arange(float(data.min(numeric_only=True)['spotsize']), max_spotsize , 0.05)

        pulsewidth = x_pulsewidth #* len(spotsize)
        energy = x_energy #* len(spotsize)
        targetthickness = x_targetthickness #* len(spotsize)

        data_to_predict = pd.DataFrame({ 
                        'energy' : energy, 
                        'pulsewidth' : pulsewidth,
                        'spotsize' : spotsize,
                        'targetthickness' : targetthickness,
                        })
        
    # data_to_predict['intensity'] = data_to_predict['energy']*4*math.log(2)/(data_to_predict['pulsewidth']*0.000000000000001*3.14*pow(data_to_predict['spotsize']*0.0001, 2))

    return data_to_predict

# Predict on a Pandas DataFrame.

#y_pred = model(pd.DataFrame(pd.DataFrame([{'spotsize':x_spotsize, 'energy':x_energy, 'pulsewidth':x_pulsewidth, 'targetthickness':x_targetthickness}])))
def get_original_datapoints(x_axis, data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, max_energy, max_pulsewidth, max_spotsize, max_targetthickness): # ehemaliges defaultargument x_axis = 'energy'

    selected = pd.DataFrame()

    x = get_data_to_predict(data, x_energy, x_pulsewidth, x_spotsize, x_targetthickness, x_axis, max_energy, max_pulsewidth, max_spotsize, max_targetthickness) # get x

    data = config.data

    print (x.shape)

    for xval in x[x_axis]:
        #df_closest = data.iloc[(data['points']-101).abs().argsort()[:1]]

        search = data.loc[(data[x_axis] >= xval-1500) & (data[x_axis] <= xval+1500)]

        closest = search.iloc[(search[x_axis]-xval).abs().argsort()[:1]]

        selected = pd.concat([selected, closest])#selected.append(closest)

        # filter by max_settings
        #i+=1
            
    # fig.add_trace(
    #     go.Scatter(
    #         x=x, y=selected['cutofenergy'], 
    #         mode='markers', customdata=selected,
    #         name='Data',
    #         # energy  pulsewidth  cutofenergy  spotsize  targetthickness     intensity Laser
    #         hovertext=[selected.pulsewidth, selected.spotsize], 
    #         hovertemplate="<br>Pulsewidth: %{customdata[1]}<br>"
    #                     + "Spotsize: %{customdata[3]}<br>"
    #                     + "Targetthickness: %{customdata[4]}",
    #         marker=dict(
    #             size=selected.spotsize,
    #             color=selected.pulsewidth,
                
    #         )
    #     )
    # ) 
    return selected #data # selected


def clear_fig ():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def smoother (window_data, window_size = 10):

    windows = window_data.rolling(window_size) 
    
    # Create a series of moving 
    # averages of each window 
    moving_averages = windows.mean() 
    
    # Convert pandas series back to list 
    moving_averages_list = moving_averages.tolist() 
    
    # Remove null entries from the list 
    final_list = moving_averages_list[window_size - 1:]

    return final_list
def get_X_with_weights(X, weight):
    X['01'] = (100-weight)/100
    X['02'] = weight/100
    return X.to_numpy()
def get_2d_predictions(mlalgorithm, data, model, x_axis='energy', test = False, slider_monotone_weight = 0):
        """_summary_

        Args:
            data (_type_): _description_
            model (_type_): _description_
            x_axis (str, optional): _description_. Defaults to 'energy'.
            test (bool): if set false def returns x and y as dataframe; if true only y. further, get_predictions will not be called

        Returns:
            DAtaFrame: containing x and y ! (only y if test=True)
        """

        if "tensorflow" in mlalgorithm or "TFLattice" in mlalgorithm:
            y = model.predict([data['energy'], data['pulsewidth'], data['spotsize'], data['targetthickness']])

        elif "stacking-monotone" in mlalgorithm:
            data['01'] = (100-slider_monotone_weight)/100
            data['02'] = slider_monotone_weight/100
            y = model.predict(data.to_numpy())
        else:
            y = model.predict(data)

        if test == False:
            #x = pd.DataFrame(get_data_to_predict(x_axis)[x_axis]) # changed 2023-12-02 02.12.2023 geändert get_2d_predictions
            x = data[x_axis]
            #st.write(x)
            df_pred_ce = pd.DataFrame(y)

            df = pd.concat([x,df_pred_ce], axis=1)
            
            df.rename(columns={ df.columns[0]: "x" , df.columns[1]: "y"}, inplace = True)
        else:
            # x = data.reset_index()
            # st.write(x.shape)
            df = pd.DataFrame(y)
            df.rename(columns={ df.columns[0]: "y" }, inplace = True)
        
        
        return df
def get_scaling_2d_predictions(mlalgorithm, data, model, x_axis='energy', test = False):
        """_summary_

        Args:
            data (_type_): _description_
            model (_type_): _description_
            x_axis (str, optional): _description_. Defaults to 'energy'.
            test (bool): if set false def returns x and y as dataframe; if true only y. further, get_predictions will not be called

        Returns:
            DAtaFrame: containing x and y ! (only y if test=True)
        """

        if "tensorflow" in mlalgorithm:
            y = model.predict([data['energy'], data['pulsewidth'], data['spotsize'], data['targetthickness']])

        elif "stacking-monotone" in mlalgorithm:
            data['01'] = (100-slider_monotone_weight)/100
            data['02'] = slider_monotone_weight/100
            y = model.predict(data.to_numpy())

        else:
            y = model.predict(data)

        if test == False:
            #x = pd.DataFrame(get_data_to_predict(x_axis)[x_axis]) # changed 2023-12-02 02.12.2023 geändert get_2d_predictions
            x = data[x_axis]
            #st.write(x)
            df_pred_ce = pd.DataFrame(y)

            df = pd.concat([x,df_pred_ce], axis=1)
            
            df.rename(columns={ df.columns[0]: "x" , df.columns[1]: "y"}, inplace = True)
        else:
            # x = data.reset_index()
            # st.write(x.shape)
            df = pd.DataFrame(y)
            df.rename(columns={ df.columns[0]: "y" }, inplace = True)
        
        
        return df
def get_prediction(data, mlalgorithm, model, monotone_weights = 0):
    if "tensorflow" in mlalgorithm or "TFLattice" in mlalgorithm:
        y = model.predict([data['energy'], data['pulsewidth'], data['spotsize'], data['targetthickness']])
    elif "stacking-monotone" in mlalgorithm:
        y = model.predict(get_X_with_weights(data, monotone_weights))
    else:
        y = model.predict(data)
    # if y is dataframe get value only
    if isinstance(y, pd.DataFrame):
        y = y.values[0]
    
    return y

def get_test_scaling_law_pred_from_train(X_test, X_train, y_train, n_mean = 3):
        y_test_scaling_law = pd.DataFrame(columns=['y_pred'])
        for testindex, testrow  in X_test.iterrows():
            # get 3 closest rows
            closest_rows = find_closest_rows(X_train, testrow['energy'], testrow['pulsewidth'], testrow['spotsize'], testrow['targetthickness'], n_mean)
            closest_rows['calculated_ec'] = closest_rows.apply(
                                                lambda closest_row: 
                                                    calc_cutofenergy_scaling_law_zimmer(
                                                        y_train.iloc[closest_row.name].cutoffenergy, 
                                                        closest_row['energy'], testrow['energy'], 
                                                        closest_row['pulsewidth'], testrow['pulsewidth'], 
                                                        closest_row['spotsize'], testrow['spotsize'], 
                                                        closest_row['targetthickness'], 
                                                        testrow['targetthickness']
                                                    ), 
                                                    axis=1
                                            )

            closest_rows['actual_ec_y'] = y_train.iloc[closest_rows.index].cutoffenergy
            
            # anzeige closest rows für debuggnig oder TODO: Thesis screenshot
            # st.write(closest_rows)

            ec_closest_mean = closest_rows['calculated_ec'].mean()
            #print ("closest_mean %2f"%(ec_closest_mean))
            y_test_scaling_law.loc[testindex] = ec_closest_mean
        return y_test_scaling_law

def get_test_scaling_law_pred_from_reference_point(X_test, reference_point):
    y_test_scaling_law = pd.DataFrame(columns=['y_pred'])
    newframe = pd.DataFrame()
    newframe['calculated_ec'] = X_test.apply(
                                    lambda X_test: 
                                        calc_cutofenergy_scaling_law_zimmer(
                                            reference_point['cutoffenergy'], 
                                            reference_point['energy'], X_test['energy'], 
                                            reference_point['pulsewidth'], X_test['pulsewidth'], 
                                            reference_point['spotsize'], X_test['spotsize'], 
                                            reference_point['targetthickness'], 
                                            X_test['targetthickness']
                                        ), 
                                        axis=1
                                )
    return (pd.DataFrame(newframe['calculated_ec']))
        