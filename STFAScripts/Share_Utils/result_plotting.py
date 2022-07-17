import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

from Essential import path_handler as ph
from Essential import global_params as gp



def history_plot(history,type_case, path=ph.get_models_history()):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')[:-6]
    path=ph.get_models_history()
    folder_name = gp.CASE[type_case] +time_now
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


def prediction_to_csv(pred):
    pass


def prediction_plot(pred):
    pass

