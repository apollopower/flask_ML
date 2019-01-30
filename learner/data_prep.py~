import numpy as np
import pandas as pd

def prep_data(data_source, csv=True):
    if csv:
        dataframe = pd.read_csv(data_source)
        return dataframe

def supervised_data(dataframe, features, target):
    data_map = {}

    data_map[x_train] = dataframe[features]
    data_map[y_train] = dataframe[target]

    return data_map

