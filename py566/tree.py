import copy
import json

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from . import base

def gain_ratio(y,x):
    y_mean = base.MeanRegressor().fit(x.to_frame(),y).predict(x.to_frame())
    base_mae = mean_absolute_error(y,y_mean)
    after_split_weight_mae = 0
    for f in x.unique():
        mask = x == f
        ysplit = y.loc[mask]
        xsplit = x.loc[mask]
        split_y_mean = base.MeanRegressor().fit(xsplit.to_frame(),ysplit).predict(xsplit.to_frame())
        split_mae = mean_absolute_error(ysplit,split_y_mean)
        after_split_weight_mae += len(ysplit)/len(y)*split_mae
    return (base_mae - after_split_weight_mae)/base_mae

class StumpRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, sample_param_here="Not using this"):
        self.sample_param_here = sample_param_here
            
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        # the code below can be modified, but I leave it here as a clue to my implementation
        predictions = []
        for value in X[self._split_column]:
            if value in self._predictions:
                predictions.append(self._predictions[value])
            else:
                predictions.append(self._mean)
            
        return np.array(predictions)