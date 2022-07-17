import datetime
import numpy as np
import tensorflow as tf

from Essential import path_handler as ph
from Training.models import *
from Share_Utils.result_plotting import *


def nf_trainer(x_train,x_val,y_train,y_val, max_row):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS NON FLUTTER START AT {time_now}')
    model, history = model_non_flutter(x_train,x_val,y_train,y_val, max_row)
    savemodel(model, history,type_case=2 , model_path=ph.get_models_non_flutter())
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS NON FLUTTER DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN NON FLUTTER MODEL IS {delta_time}')

    return model, history, delta_time



def nf_inferencer(mach, vf, num_model=1,path=ph.get_models_non_flutter(), type_case=2):
    model, history = load_model(path, type_case, num_model)
    # Plot and save Histories`plot
    history_plot(history,mach, vf, type_case=type_case, path=ph.get_models_history())

    #Predict 
    pred = predict_non_class(model, mach, vf)
    prediction_to_csv(pred,mach, vf, type_case=type_case)
    prediction_plot(pred,mach, vf, type_case=type_case)

    return pred