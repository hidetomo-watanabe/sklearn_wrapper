import numpy as np


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
    before = [chr(ord('a') + i) for i in range(26)]
    after = list(range(26))
    for df in [train_df, test_df]:
        df['ord_3_label'] = df['ord_3'].fillna(-1).replace(
            before, after).astype('float32')
    return train_df, test_df


def add_ord_4_label(train_df, test_df):
    before = [chr(ord('a') + i).upper() for i in range(26)]
    after = list(range(26))
    for df in [train_df, test_df]:
        df['ord_4_label'] = df['ord_4'].replace(
            before, after).astype('float32')
    return train_df, test_df


def add_splited_ord_5(train_df, test_df):
    for df in [train_df, test_df]:
        df['ord_5_1'] = df['ord_5'].str[0]
        df['ord_5_2'] = df['ord_5'].str[1]
    return train_df, test_df


def add_ord_5_1_label(train_df, test_df):
    before = [chr(ord('a') + i) for i in range(26)]
    before += [chr(ord('a') + i).upper() for i in range(26)]
    after = list(range(52))
    for df in [train_df, test_df]:
        df['ord_5_1_label'] = df['ord_5_1'].fillna(-1).replace(
            before, after).astype('float32')
    return train_df, test_df


def add_ord_5_2_label(train_df, test_df):
    before = [chr(ord('a') + i) for i in range(26)]
    before += [chr(ord('a') + i).upper() for i in range(26)]
    after = list(range(52))
    for df in [train_df, test_df]:
        df['ord_5_2_label'] = df['ord_5_2'].fillna(-1).replace(
            before, after).astype('float32')
    return train_df, test_df


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


def translate_proba2submission(Y_pred, Y_pred_proba):
    Y_pred_proba = Y_pred_proba.drop('target_0', axis=1)
    Y_pred_proba = Y_pred_proba.rename(columns={'target_1': 'target'})
    return Y_pred, Y_pred_proba
