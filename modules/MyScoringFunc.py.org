import numpy as np
from sklearn.metrics.scorer import make_scorer


def _get_my_loss_err(Y, Y_pred):
    err = np.sum(np.abs(Y - Y_pred))
    return err


def get_my_scorer():
    return make_scorer(_get_my_loss_err, greater_is_better=False)
