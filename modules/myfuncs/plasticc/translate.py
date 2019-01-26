import numpy as np


def translate_target(train_df, test_df):
    #######################################
    # rename target to class
    #######################################
    train_df = train_df.rename(columns={'target': 'class'})
    return train_df, test_df


def translate_mjd(train_df, test_df):
    #######################################
    # mjd_diff = mjd_max - mjd_min
    #######################################
    for df in [train_df, test_df]:
        df['mjd_diff'] = df['mjd_max'] - df['mjd_min']
        del df['mjd_max']
        del df['mjd_min']
    return train_df, test_df


def translate_flux(train_df, test_df):
    #######################################
    # flux_diff = flux_max - flux_min
    #######################################
    for df in [train_df, test_df]:
        df['flux_diff'] = df['flux_max'] - df['flux_min']
        df['flux_diff2'] = df['flux_diff'] / df['flux_mean']
    return train_df, test_df


def add_class_99(Y_pred, Y_pred_proba):
    #######################################
    # add average proba to class_99
    #######################################
    n_gal = len(Y_pred_proba.columns) - 1
    for column in Y_pred_proba.columns:
        if column == 'object_id':
            continue
        Y_pred_proba[column] = Y_pred_proba[column] * n_gal / (n_gal + 1)
    Y_pred_proba['class_99'] = np.array(
        [[1 / (n_gal + 1)]] * Y_pred_proba.shape[0])
    return Y_pred, Y_pred_proba
