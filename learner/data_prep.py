import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def create_supervised_dataset(data_path, features, target, csv=True):
    if not csv:
        return "Error, data not CSV" # Hack solution for now
    else:
        df = pd.read_csv(data_path)
    data_map = {
            'x_train': df[features],
            'y_train': df[target]
            }
    return data_map

def linear_regression(data_map):
   linear_reg = LinearRegression()
   linear_reg.fit(data_map['x_train'], data_map['y_train'])
   y_predict = linear_reg.predict(data_map['x_train'])
   print("Trained!")
   print("-" * 80)
   print("Mean Absolute Error {}".format(metrics.mean_absolute_error(data_map['y_train'], y_predict)))

