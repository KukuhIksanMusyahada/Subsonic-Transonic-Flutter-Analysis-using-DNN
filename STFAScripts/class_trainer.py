import datetime
import numpy as np
import tensorflow as tf

from Essential import path_handler as ph
from Training.models import model_classification, savemodel


def classification_trainer(x_train,x_val,y_train,y_val):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING CLASSIFICATION FLUTTER START AT {time_now}')
    model, history = model_classification(x_train,x_val,y_train,y_val)
    savemodel(model, history,type_case=4 , model_path=ph.get_models_classification())
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING CLASSIFICATION DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN CLASSIFICATION MODEL IS {delta_time}')

    return model, history

    