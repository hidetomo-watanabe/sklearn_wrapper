from functools import reduce
from operator import mul

import numpy as np

import pandas as pd

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

    @classmethod
    def sample_like(self, target, frac):
        if isinstance(target, pd.DataFrame):
            return target.sample(
                frac=frac, replace=False, random_state=42)
        elif isinstance(target, np.ndarray):
            if target.ndim == 1:
                _target = target
            else:
                _target = target.reshape(
                    (-1, reduce(mul, target.shape[1:])))
            df = pd.DataFrame(_target)
            _target = df.sample(
                frac=frac, replace=False, random_state=42).to_numpy()
            if target.ndim == 1:
                return _target
            else:
                return _target.reshape(-1, *target.shape[1:])
        return target
