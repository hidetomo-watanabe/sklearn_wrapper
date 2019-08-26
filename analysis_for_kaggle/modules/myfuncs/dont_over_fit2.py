import numpy as np


def translate_target2int(train_df, test_df):
    train_df['target'] = train_df['target'].astype(np.int64)
    return train_df, test_df


def translate_proba2submission(Y_pred, Y_pred_proba):
    Y_pred_proba = Y_pred_proba.drop('target_0', axis=1)
    Y_pred_proba = Y_pred_proba.rename(columns={'target_1': 'target'})
    return Y_pred, Y_pred_proba


def translate_pred2max1(Y_pred, Y_pred_proba):
    Y_pred['target'] /= np.max(Y_pred['target'])
    return Y_pred, Y_pred_proba
