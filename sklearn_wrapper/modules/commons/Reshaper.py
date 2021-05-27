from keras.utils.np_utils import to_categorical


class Reshaper(object):
    def __init__(self, shape, is_categorical=False):
        self.shape = shape
        self.is_categorical = is_categorical

    def fit(self, X, y=None):
        return self

    def fit_resample(self, X, y=None):
        self.fit(X, y)
        if len(self.shape) > 1:
            X = X.reshape(-1, *self.shape)
        if self.is_categorical:
            y = to_categorical(y)
        return X, y
