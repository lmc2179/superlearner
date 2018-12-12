from superlearner.model import SuperLearnerRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
X, y = load_boston(return_X_y=True)
sl = SuperLearnerRegressor([LinearRegression(), Ridge(), Lasso()], k_folds=10)
print(np.mean(cross_val_score(LinearRegression(), X, y, scoring='neg_mean_squared_error', cv=10)))
print(np.mean(cross_val_score(Ridge(), X, y, scoring='neg_mean_squared_error', cv=10)))
print(np.mean(cross_val_score(Lasso(), X, y, scoring='neg_mean_squared_error', cv=10)))
print(np.mean(cross_val_score(sl, X, y, scoring='neg_mean_squared_error', cv=10)))
