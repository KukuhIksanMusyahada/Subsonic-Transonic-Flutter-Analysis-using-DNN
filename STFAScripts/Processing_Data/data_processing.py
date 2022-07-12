import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from Essential import path_handler as ph
from Essential import global_params as gp
from Processing_Data import get_data as gd


def create_training_array(path):
    label_array= list()
    input_array = list()
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path, nrows= gp.NROWS, usecols=[1,2,3,4]).to_numpy()
            input_array.append([gd.extract_mach_and_vf(file)])
            label_array.append(df)
    return np.array(input_array), np.array(label_array)

# def split_training_array(input, label, train_ratio, shuffle= False,):
#     X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= shuffle)
#     return X_train, X_val, y_train, y_val



def process_data_classification(file='Flutter_Classification_Data.csv',path= ph.get_flutter_class_data(), train_ratio= gp.TRAIN_RATIO):
    input = pd.read_csv(os.path.join(path, file), usecols=['Mach', 'Vf']).to_numpy()
    label = pd.read_csv(os.path.join(path, file), usecols=['Flutter']).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= True)
    return X_train, X_val, y_train, y_val


def process_data_flutter(path = ph.get_flutter_data(),train_ratio= gp.TRAIN_RATIO):
    input, label = create_training_array(path)
    label = label * 10000
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= False)
    return X_train, X_val, y_train, y_val

def process_data_non_flutter(path= ph.get_non_flutter_data(),train_ratio= gp.TRAIN_RATIO):
    input, label = create_training_array(path)
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= False)
    return X_train, X_val, y_train, y_val


def process_data_transonic(path= ph.get_transonic_data(),train_ratio= gp.TRAIN_RATIO):
    input, label = create_training_array(path)
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= False)
    return X_train, X_val, y_train, y_val

