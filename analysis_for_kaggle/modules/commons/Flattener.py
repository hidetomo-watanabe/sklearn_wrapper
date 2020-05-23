from functools import reduce
from operator import mul


class Flattener(object):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_resample(self, X, y=None):
        self.fit(X, y)
        if X.ndim > 1:
            X = X.reshape(
                (-1, reduce(mul, X.shape[1:])))
        return X, y
