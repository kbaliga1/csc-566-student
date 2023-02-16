import numpy as np
import pandas as pd

from sklearn.utils import resample

from sklearn.base import BaseEstimator, RegressorMixin

from . import base

def get_learner_example():
    return base.MeanRegressor()

def boostrap_sample(X,y):
    X, y = resample(X, y)
    return X,y

class BaggingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, ntrees=10, get_learner_func=get_learner_example,seed=42):
        self._get_learner_func = get_learner_func
        self._ntrees = ntrees
        self._seed = seed
            
    def fit(self, X, y):
        np.random.seed(self._seed)

        self._trees = []
        for i in range(self._ntrees):
            # Your solution here
            sampleX, sampley = boostrap_sample(X,y)
            self._trees.append(self._get_learner_func().fit(sampleX,sampley))

        return self
    
    def predict(self, X):
        # the code below can be modified, but I leave it here as a clue to my implementation
        tree_predictions = []
        for j in range(len(self._trees)):
            tree = self._trees[j]
            tree_predictions.append(tree.predict(X).tolist())
        return np.array(pd.DataFrame(tree_predictions).mean().values.flat)