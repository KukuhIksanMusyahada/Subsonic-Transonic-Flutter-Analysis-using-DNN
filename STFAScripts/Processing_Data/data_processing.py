import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from Essential import path_handler as ph
from Essential import global_params as gp
from Processing_Data import get_data as gd

def process_data_classification(file='Flutter_Classification_Data.csv',path= ph.get_flutter_class_data(), train_ratio= 0.9):
    input = pd.read_csv(os.path.join(path, file), usecols=['Mach', 'Vf']).to_numpy()
    label = pd.read_csv(os.path.join(path, file), usecols=['Flutter']).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= True)
    return X_train, X_val, y_train, y_val
    

def process_data_flutter():
    pass


def process_data_non_flutter():
    pass


def process_data_transonic():
    pass

