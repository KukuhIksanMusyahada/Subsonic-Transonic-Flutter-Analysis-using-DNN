import pandas as pd

from Essential import path_handler as ph
from Essential import global_params as gp

def get_df(path):
    try:
        df = pd.read_csv(path, usecols=gp.COLUMNS1)
    except ValueError:
        df = pd.read_csv(path, usecols=gp.COLUMNS2)

    return df
