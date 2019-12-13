import numpy as np
import pandas as pd


def add_day_cyclic(train_df, test_df):
    for df in [train_df, test_df]:
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
    return train_df, test_df


def add_month_cyclic(train_df, test_df):
    for df in [train_df, test_df]:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return train_df, test_df


def add_bin_3_label(train_df, test_df):
    for df in [train_df, test_df]:
        df['bin_3_label'] = df['bin_3'].replace({
            'T': 1,
            'F': 0,
        })
    return train_df, test_df


def add_bin_4_label(train_df, test_df):
    for df in [train_df, test_df]:
        df['bin_4_label'] = df['bin_4'].replace({
            'Y': 1,
            'N': 0,
        })
    return train_df, test_df


def add_nom_5_freq(train_df, test_df):
    sum_df = pd.concat([train_df, test_df], sort=False)
    freqs = (sum_df.groupby('nom_5').size()) / len(sum_df)
    for df in [train_df, test_df]:
        df['nom_5_freq'] = df['nom_5'].apply(lambda x: freqs[x])
    return train_df, test_df


def add_nom_6_freq(train_df, test_df):
    sum_df = pd.concat([train_df, test_df], sort=False)
    freqs = (sum_df.groupby('nom_6').size()) / len(sum_df)
    for df in [train_df, test_df]:
        df['nom_6_freq'] = df['nom_6'].apply(lambda x: freqs[x])
    return train_df, test_df


def add_nom_7_freq(train_df, test_df):
    sum_df = pd.concat([train_df, test_df], sort=False)
    freqs = (sum_df.groupby('nom_7').size()) / len(sum_df)
    for df in [train_df, test_df]:
        df['nom_7_freq'] = df['nom_7'].apply(lambda x: freqs[x])
    return train_df, test_df


def add_nom_8_freq(train_df, test_df):
    sum_df = pd.concat([train_df, test_df], sort=False)
    freqs = (sum_df.groupby('nom_8').size()) / len(sum_df)
    for df in [train_df, test_df]:
        df['nom_8_freq'] = df['nom_8'].apply(lambda x: freqs[x])
    return train_df, test_df


def add_nom_9_freq(train_df, test_df):
    sum_df = pd.concat([train_df, test_df], sort=False)
    freqs = (sum_df.groupby('nom_9').size()) / len(sum_df)
    for df in [train_df, test_df]:
        df['nom_9_freq'] = df['nom_9'].apply(lambda x: freqs[x])
    return train_df, test_df


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


def add_ord_5_label(train_df, test_df):
    mapping = sorted(list(set(train_df['ord_5'].values)))
    mapping = dict(zip(mapping, range(len(mapping))))
    for df in [train_df, test_df]:
        df['ord_5_label'] = df['ord_5'].apply(lambda x: mapping[x]).astype(int)
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
