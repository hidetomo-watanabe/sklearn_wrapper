import numpy as np


def add_image_id(train_df, test_df):
    for df in [train_df, test_df]:
        df['ImageId'] = np.arange(1, len(df) + 1)
    return train_df, test_df


def translate_label2upper(Y_pred, Y_pred_proba):
    Y_pred = Y_pred.rename(columns={'label': 'Label'})
    return Y_pred, Y_pred_proba
