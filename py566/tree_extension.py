import copy
import json

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from . import base


def gain_ratio(y, x):
    y_mean = base.MeanRegressor().fit(x.to_frame(), y).predict(x.to_frame())
    base_mae = mean_absolute_error(y, y_mean)
    after_split_weight_mae = 0
    for f in x.unique():
        mask = x == f
        ysplit = y.loc[mask]
        xsplit = x.loc[mask]
        split_y_mean = base.MeanRegressor().fit(xsplit.to_frame(), ysplit).predict(xsplit.to_frame())
        split_mae = mean_absolute_error(ysplit, split_y_mean)
        after_split_weight_mae += len(ysplit) / len(y) * split_mae
    return (base_mae - after_split_weight_mae) / base_mae


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, depth=3):
        self.depth = depth

    def fit(self, X, y):
        # find which column gives the best gain ratio
        self._split_column = ""
        gain = -1
        for label in X.columns:
            curGain = gain_ratio(y, X[label])
            if curGain > gain:
                gain = curGain
                self._split_column = label

        #for each of the values in split column either create a prediction or create a new tree
        r,c = X.shape
        if self._split_column == "":
            self._predictions = {}
        elif self.depth == 1:
            self._predictions = pd.concat([X, y], axis=1).groupby(self._split_column)[y.name].mean()
        else:
            self._predictions = {}
            for value in X[self._split_column].unique():
                #for each unique value in the current level, filter out values and create new tree
                tree = DecisionTreeRegressor(depth=self.depth-1)
                mask = X[self._split_column] == value
                newX = X[mask]
                newy = y[mask]
                tree.fit(newX,newy)
                self._predictions[value] = tree

        # calculate default value for y
        self._mean = y.mean()

        return self

    def predict(self, X):
        # the code below can be modified, but I leave it here as a clue to my implementation
        predictions = []
        if isinstance(X,pd.DataFrame):
            for index, row in X.iterrows():
                if row[self._split_column] in self._predictions:
                    if self.depth == 1:
                        predictions.append(self._predictions[row[self._split_column]])
                    else:
                        predictions.append(self._predictions[row[self._split_column]].predict(row))
                else:
                    predictions.append(self._mean)
        elif isinstance(X,pd.Series):
            if self._split_column == "":
                return self._mean
            elif X[self._split_column] in self._predictions:
                if self.depth == 1:
                    return self._predictions[X[self._split_column]]
                else:
                    return self._predictions[X[self._split_column]].predict(X)
            else:
                return self._mean

        return np.array(predictions)