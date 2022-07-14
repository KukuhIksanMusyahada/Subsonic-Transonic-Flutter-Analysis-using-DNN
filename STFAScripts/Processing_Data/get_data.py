from ctypes import sizeof
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

from Essential import path_handler as ph
from Essential import global_params as gp


def extract_mach_and_vf(file: str):
    pattern = r'M_([0-9\.]*)_VF_([0-9\.]*).csv'
    result  = re.match(pattern, file)

    return float(result.group(1)), float(result.group(2))

def get_df(path, flutter_test=False, usecols1= gp.COLUMNS1, usecols2=gp.COLUMNS2):
    if flutter_test == False:
        try:
            df = pd.read_csv(path, usecols=gp.COLUMNS1, engine='python')
            
        except ValueError:
            df = pd.read_csv(path, usecols=gp.COLUMNS2,engine='python')
        
    else:
        col = [['plunge(airfoil)'], ['plunge_airfoil']]
        try:
            df = pd.read_csv(path, usecols= col[0],engine='python').to_numpy()
        except ValueError:
            df = pd.read_csv(path, usecols= col[1],engine='python').to_numpy()

    return df

def gradien(array):
    ''' Calculate the gradien of an array on each point. Assuming that each 
    horizontal axis have the same step so the gradien is just the delta of its current value
    with the  next step'''
    grad = []
    for row in range(array.shape[0]-1):
        delta = array[row+1]- array[row]
        grad.append(delta)
    return grad

def find_turn_point(grad):
    before_sign = -1
    pos_index = []
    for index, elem in enumerate(grad):
        if elem !=0:
            sign = elem/ abs(elem)
        else:
            sign = before_sign
        if sign < before_sign:
            pos_index.append(index)
        before_sign = sign

    return pos_index

def divergence_test(array, index, wait =3):
    count = 0
    val_before= array[index[0]]
    divergen = 0
    for i in index:
        value = array[i]
        if value > val_before:
            count +=1
        if count == wait:
            divergen= 1
        val_before = value
    return divergen


def split_cases(file,path):
    mach, vf = extract_mach_and_vf(file)
    df = get_df(path, flutter_test=False)
    arr = get_df(path, flutter_test= True)
    grad= gradien(arr)
    turn_point = find_turn_point(grad)
    divergen = divergence_test(arr, turn_point)
    df = df[(np.abs(stats.zscore(df)) < 1.5).all(axis=1)]
    if df.shape[0]==0:
        row_size = 999
    else:
        row_size = df.shape[0]
    if divergen== 0:
        df.to_csv(os.path.join(ph.get_non_flutter_data(), file))
    elif divergen ==1:
        df.to_csv(os.path.join(ph.get_flutter_data(), file))
    return mach, vf, divergen, row_size
        



def scan(path= ph.get_raw_data()):
    Mach= []
    Vf = []
    Flutter = []
    row_count = 999
    row_count_transonic = 999
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if dir == 'M_0.9':
            for file in os.listdir(dir_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(dir_path, file)
                    df = get_df(file_path, flutter_test=False)
                    df = df[(np.abs(stats.zscore(df)) < 1.5).all(axis=1)]
                    if df.shape[0]<row_count_transonic:
                        row_count_transonic = df.shape[0]
                    df.to_csv(os.path.join(ph.get_transonic_data(), file))
        else:
            for file in os.listdir(dir_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(dir_path, file)
                    mach,vf, divergen, size = split_cases(file, file_path)
                    if size < row_count:
                        row_count = size
                    Mach.append(mach)
                    Vf.append(vf)
                    Flutter.append(divergen)
    data = np.array([Mach, Vf, Flutter]).T
    df_clf = pd.DataFrame(data, columns=['Mach', 'Vf', 'Flutter'])
    df_clf.to_csv(os.path.join(ph.get_flutter_class_data(),'Flutter_Classification_Data.csv'))
    return row_count, row_count_transonic
                
