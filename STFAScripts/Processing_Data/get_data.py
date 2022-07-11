import re
import pandas as pd

from Essential import path_handler as ph
from Essential import global_params as gp


def extract_mach_and_vf(file: str):
    pattern = r'M_([0-9\.]*)_VF_([0-9\.]*).csv'
    result  = re.match(pattern, file)

    return float(result.group(1)), float(result.group(2))

def get_df(path):
    try:
        df = pd.read_csv(path, usecols=gp.COLUMNS1)
    except ValueError:
        df = pd.read_csv(path, usecols=gp.COLUMNS2)

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


def split_cases(path= ph.get_raw_data()):
    mach= []
    vf = []
    divergence = []
    