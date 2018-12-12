from scipy.optimize import nnls

class NonNegativeLeastSquares(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        self.coef_ = nnls(X, y)

    def predict(self, X):
        return X * self.coef_
