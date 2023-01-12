import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)


def make_trees(X,y,ntrees=100,max_depth=10):
    trees = []
    for i in range(ntrees):
        # Your solution here
        pass
        
    return trees

def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)