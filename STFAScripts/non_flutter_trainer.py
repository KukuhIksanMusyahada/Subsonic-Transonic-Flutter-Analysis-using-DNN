import datetime
import numpy as np
import tensorflow as tf

from Essential import path_handler as ph
from Training.models import model_non_flutter, savemodel


def nf_trainer(x_train,x_val,y_train,y_val, max_row):
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS NON FLUTTER START AT {time_now}')
    model, history = model_non_flutter(x_train,x_val,y_train,y_val, max_row)
    savemodel(model, history,type_case=1 , model_path=ph.get_models_non_flutter())
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING MODELS NON FLUTTER DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN NON FLUTTER MODEL IS {delta_time}')

    return model, history, delta_time