from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, KFold
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.base import BaseEstimator

class SuperLearnerRegressor(BaseEstimator):
    def __init__(self, base_learners, k_folds=3):
        self.base_learners = base_learners
        self.meta_learner = Lasso(alpha=0, positive=True)
        self.k_folds = k_folds
        self.trained_learners_ = None
        self.trained_meta_learner_ = None

    def fit(self, X, y):
        seed = np.random.randint(-1e6, 1e6)
        y_pred_learners = [cross_val_predict(l, X, y, cv=KFold(self.k_folds, random_state=seed)) for l in self.base_learners]
        self.trained_learners_ = [deepcopy(l).fit(X, y) for l in self.base_learners]
        Z = np.vstack(y_pred_learners).T
        self.trained_meta_learner_ = deepcopy(self.meta_learner)
        self.trained_meta_learner_.fit(Z, y)
        
    def predict(self, X):
        y_pred_learners = [l.predict(X) for l in self.trained_learners_]
        Z = np.vstack(y_pred_learners).T
        return self.trained_meta_learner_.predict(Z)
