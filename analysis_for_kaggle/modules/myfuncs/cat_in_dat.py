import numpy as np
import pandas as pd


def add_ord_1_label(train_df, test_df):
    for df in [train_df, test_df]:
        df['ord_1_label'] = df['ord_1'].replace({
            'Novice': 0,
            'Contributor': 1,
            'Expert': 2,
            'Master': 3,
            'Grandmaster': 4,
        })
    return train_df, test_df


def add_ord_2_label(train_df, test_df):
    for df in [train_df, test_df]:
        df['ord_2_label'] = df['ord_2'].replace({
            'Freezing': 0,
            'Cold': 1,
            'Warm': 2,
            'Hot': 3,
            'Boiling Hot': 4,
            'Lava Hot': 5,
        })
    return train_df, test_df


def add_ord_3_label(train_df, test_df):
    for df in [train_df, test_df]:
        df['ord_3_label'] = df['ord_3'].replace(
            [chr(ord('a') + i) for i in range(26)],
            list(range(26))
        ).astype(int)
    return train_df, test_df


def add_ord_4_label(train_df, test_df):
    for df in [train_df, test_df]:
        df['ord_4_label'] = df['ord_4'].replace(
            [chr(ord('a') + i).upper() for i in range(26)],
            list(range(26))
        ).astype(int)
    return train_df, test_df


def split_ord_5(train_df, test_df):
    for df in [train_df, test_df]:
        df['ord_5_1'] = df['ord_5'].str[0]
        df['ord_5_2'] = df['ord_5'].str[1]
        df.drop('ord_5', axis=1, inplace=True)
    return train_df, test_df


def translate_proba2submission(Y_pred, Y_pred_proba):
    Y_pred_proba = Y_pred_proba.drop('target_0', axis=1)
    Y_pred_proba = Y_pred_proba.rename(columns={'target_1': 'target'})
    return Y_pred, Y_pred_proba
