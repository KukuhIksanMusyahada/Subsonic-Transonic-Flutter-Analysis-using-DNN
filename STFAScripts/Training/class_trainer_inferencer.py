import datetime
import numpy as np
import tensorflow as tf

from Essential import path_handler as ph
from Training.models import *


def classification_trainer(x_train,x_val,y_train,y_val):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING CLASSIFICATION FLUTTER START AT {time_now}')
    model, history = model_classification(x_train,x_val,y_train,y_val)
    savemodel(model, history,type_case=0 , model_path=ph.get_models_classification())
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING CLASSIFICATION DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN CLASSIFICATION MODEL IS {delta_time}')

    return model, history, delta_time



def classification_inferencer(mach, vf,type_case=0, num_model=1):
    model, history = load_model(ph.get_models_classification(), type_case=type_case,num_model=num_model)
    # Plot and save Histories`plot

    #Predict 
    prediction = predict_class(model, mach, vf)

    return prediction