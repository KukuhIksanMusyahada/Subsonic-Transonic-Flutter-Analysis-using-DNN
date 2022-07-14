import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from Essential import path_handler as ph
from Essential import global_params as gp
from Processing_Data import get_data as gd


def create_training_array(path, max_row):
    label_array= list()
    input_array = list()
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path, nrows= max_row, usecols=[1,2,3,4]).to_numpy()
            if df.shape[0]==0:
                continue
            result = gd.extract_mach_and_vf(file)
            mach = result[0]
            vf = result[1]
            input_array.append([mach, vf])
            label_array.append(df)
    # label_array = np.concatenate(label_array)
    return np.array(input_array), np.array(label_array)





# Normalization




# Find Weight Value
def find_weight_value(arr):
    new_arr = list()
    for col in range(arr.shape[1]):
        new_arr.append(1/np.mean(arr[:, col]))
    return np.array(new_arr)

def scaler(arr, multiplier):
    new_arr = list()
    for col in range(arr.shape[1]):
        new_arr.append(np.multiply(arr[:,col], multiplier[col]))
    return np.array(new_arr)


def process_data_classification(file='Flutter_Classification_Data.csv',path= ph.get_flutter_class_data(), train_ratio= gp.TRAIN_RATIO):
    input = pd.read_csv(os.path.join(path, file), usecols=['Mach', 'Vf']).to_numpy()
    label = pd.read_csv(os.path.join(path, file), usecols=['Flutter']).to_numpy() 
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= True)
    return X_train, X_val, y_train, y_val


def process_data_flutter(max_row, path = ph.get_flutter_data(),train_ratio= gp.TRAIN_RATIO):
    input, label = create_training_array(path, max_row)
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= False) 
    return X_train, X_val, y_train, y_val

def process_data_non_flutter(max_row,path= ph.get_non_flutter_data(),train_ratio= gp.TRAIN_RATIO):
    input, label = create_training_array(path,max_row)
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= False)
    return X_train, X_val, y_train, y_val


def process_data_transonic(max_row, path= ph.get_transonic_data(),train_ratio= gp.TRAIN_RATIO):
    input, label = create_training_array(path,max_row)
    X_train, X_val, y_train, y_val = train_test_split(input,label, train_size= train_ratio, shuffle= False)
    return X_train, X_val, y_train, y_val



## DEPRECIATED ##
# def find_weight_value(arr):
#     multiplier = list()
#     for batch in range(arr.shape[0]):
#         batch_multiplier = list()
#         for col in range(arr.shape[2]):
#             batch_multiplier.append(1/np.median(arr[batch,:,col]))
#         multiplier.append(batch_multiplier)
#     return np.array(multiplier)

# def scaler(arr, multipliers):
#     new_arr = list()
#     for x, y in zip(arr, multipliers):
#         z = np.multiply(x, y)
#         new_arr.append(z)
#     return np.array(new_arr)

# def invers_scaler(arr, multipliers):
#     new_arr = list()
#     for x, y in zip(arr, multipliers):
#         z = np.divide(x, y)
#         new_arr.append(z)
#     return np.array(new_arr)