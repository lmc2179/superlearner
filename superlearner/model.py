from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
import numpy as np
from copy import deepcopy

class SuperLearnerRegressor(object):
    def __init__(self, base_learners, k_folds=3):
        self.base_learners = base_learners
        self.k_folds = k_folds
        self.trained_learners_ = None
        self.trained_meta_learner_ = None

    def ffit(self, X, y):
        pass

    def predict(self, X):
        pass
