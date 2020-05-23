import numpy as np

import pandas as pd

import scipy.sparse as sp


if 'Flattener' not in globals():
    from .Flattener import Flattener
if 'Reshaper' not in globals():
    from .Reshaper import Reshaper


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
            _target, _ = Flattener().fit_resample(target)
            df = pd.DataFrame(_target)
            _target = df.sample(
                frac=frac, replace=False, random_state=42).to_numpy()
            _target, _ = Reshaper(target.shape[1:]).fit_resample(_target)
            return _target
        return target
