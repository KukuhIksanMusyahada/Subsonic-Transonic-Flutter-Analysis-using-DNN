import datetime
import numpy as np
import tensorflow as tf

from Essential import path_handler as ph
from Training.models import model_transonic, savemodel


def transonic_trainer(x_train,x_val,y_train,y_val, max_row):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS TRANSONIC START AT {time_now}')
    model, history = model_transonic(x_train,x_val,y_train,y_val, max_row)
    savemodel(model, history,type_case=3 , model_path=ph.get_models_transonic())
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS TRANSONIC DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN TRANSONIC MODEL IS {delta_time}')

    return model, history, delta_time



def transonic_inferencer(mach, vf):
    pass