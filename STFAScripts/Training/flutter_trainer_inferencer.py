import datetime
import numpy as np
import tensorflow as tf

from Essential import path_handler as ph
from Training.models import *

def f_trainer(x_train,x_val,y_train,y_val, max_row):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS FLUTTER START AT {time_now}')
    model, history = model_flutter(x_train,x_val,y_train,y_val, max_row)
    savemodel(model, history,type_case=1 , model_path=ph.get_models_flutter())
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS FLUTTER DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN FLUTTER MODEL IS {delta_time}')

    return model, history, delta_time


def f_inferencer(mach, vf, num_model,path=ph.get_models_flutter(), type_case=3):
    model, history = load_model(path, num_model, type_case)
    # Plot and save Histories`plot

    #Predict 
    prediction = predict_non_class(model, mach, vf)

    return prediction