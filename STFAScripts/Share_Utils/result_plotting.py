import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

from Essential import path_handler as ph
from Essential import global_params as gp



def history_plot(history, mach, vf, type_case, path=ph.get_models_history()):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H-%M-%S')
    path=ph.get_models_history()
    folder_name = gp.CASE[type_case] +str(mach)+ str(vf)+time_now
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    for key in history.keys():
            file_name = str(key)+'.png'
            file_path = os.path.join(folder_path, file_name)
            y = history[key]
            ax.clear()
            line1, = ax.plot(y)
            plt.title(key)
            plt.xlabel('Epochs')
            plt.ylabel(key)
            plt.savefig(file_path)
            
            print(f'History Figure saved in {os.path.abspath(folder_path)}')


def prediction_to_csv(pred,mach, vf, type_case, path=ph.get_models_prediction()):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H-%M-%S')
    folder_name = gp.CASE[type_case] +'-'+str(mach)+'-' + str(vf) +'-'+ time_now
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    pred_df = pd.DataFrame(pred, columns=['CL_Pred', 'CD_Pred', 'Plunge_Pred', 'Pitch_Pred'])
    pred_df.to_csv(os.path.join(folder_path,'pred_df.csv'))


def prediction_plot(pred, mach, vf, type_case, path=ph.get_models_prediction()):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H-%M-%S')
    folder_name = gp.CASE[type_case] +'-'+str(mach)+'-' + str(vf) +'-'+ time_now
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    figure, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel('Time Step')
    for col in range(pred.shape[1]):
        if col == 0:
            plt.title('CL')
            plt.ylabel('CL')
            file_name = 'CL.png'
            file_path = os.path.join(folder_path,file_name)
        elif col ==1:
            plt.title('CD')
            plt.ylabel('CD')
            file_name = 'CD.png'
            file_path = os.path.join(folder_path,file_name)
        elif col ==2:
            plt.title('Plunge')
            plt.ylabel('Plunge')
            file_name = 'Plunge.png'
            file_path = os.path.join(folder_path,file_name)
        elif col ==3:
            plt.title('Pitch')
            plt.ylabel('Pitch')
            file_name = 'Pitch.png'
            file_path = os.path.join(folder_path,file_name)
        else:
            print('Max features are 4')
        y = pred[:, col]
        ax.clear()
        line1, = ax.plot(y)
        plt.savefig(file_path)
        print(f'Prediction Figure saved in {os.path.abspath(folder_path)}')


