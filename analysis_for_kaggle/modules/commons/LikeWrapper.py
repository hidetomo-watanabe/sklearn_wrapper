import scipy.sparse as sp


class LikeWrapper(object):
    def __init__(self):
        pass

    @classmethod
    def ravel_like(self, ndarray):
        if ndarray.ndim > 1 and ndarray.shape[1] == 1:
            return ndarray.ravel()
        return ndarray

    @classmethod
    def toarray_like(self, target):
        if isinstance(target, sp.csr.csr_matrix):
            return target.toarray()
        return target
