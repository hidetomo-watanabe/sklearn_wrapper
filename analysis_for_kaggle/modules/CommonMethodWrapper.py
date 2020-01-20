class CommonMethodWrapper(object):
    def __init__(self):
        pass

    @classmethod
    def ravel_like(self, ndarray):
        if ndarray.ndim > 1 and ndarray.shape[1] == 1:
            return ndarray.ravel()
        return ndarray
