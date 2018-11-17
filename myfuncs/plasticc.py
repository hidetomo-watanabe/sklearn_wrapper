import numpy as np


def translate_target(dfs, train_df):
    #######################################
    # rename target to class
    #######################################
    for df in dfs:
        if 'target' not in df.columns:
            continue
        df['class'] = df['target']
        del df['target']
    return dfs


def translate_mjd(dfs, train_df):
    #######################################
    # mjd_diff = mjd_max - mjd_min
    #######################################
    for df in dfs:
        df['mjd_diff'] = df['mjd_max'] - df['mjd_min']
        del df['mjd_max']
        del df['mjd_min']
    return dfs


def translate_flux(dfs, train_df):
    #######################################
    # flux_diff = flux_max - flux_min
    #######################################
    for df in dfs:
        df['flux_diff'] = df['flux_max'] - df['flux_min']
        df['flux_diff2'] = df['flux_diff'] / df['flux_mean']
    return dfs


def add_class99(Y_pred, Y_pred_proba):
    #######################################
    # add average proba to class99
    #######################################
    n_gal = Y_pred_proba.shape[1]
    Y_pred_proba = Y_pred_proba * n_gal / (n_gal + 1)
    class99 = np.array([[1 / (n_gal + 1)]] * Y_pred_proba.shape[0])
    Y_pred_proba = np.hstack((Y_pred_proba, class99))
    return Y_pred, Y_pred_proba


if __name__ == '__main__':
    pass
