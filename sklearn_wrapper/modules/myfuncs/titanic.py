from keras.layers.core import Dense
from keras.models import Sequential

import numpy as np

import pandas as pd

import torch.nn.functional as F
from torch import nn


def translate_title(train_df, test_df):
    #######################################
    # name => title
    #######################################
    for df in [train_df, test_df]:
        df['Title'] = 'Other'
        for i, val in enumerate(df['Name'].to_numpy()):
            title = val.split(',')[1].split('.')[0].strip()
            if title in ['Mr', 'Mrs', 'Miss', 'Master']:
                df['Title'].to_numpy()[i] = title
    return train_df, test_df


def translate_age(train_df, test_df):
    #######################################
    # no age => median grouped by title, pclass
    # binning 10
    #######################################
    _2median = train_df.groupby(['Title', 'Pclass'])['Age'].median()
    for df in [train_df, test_df]:
        for i, val in enumerate(df['Age'].to_numpy()):
            if not np.isnan(val):
                continue

            _t = df['Title'].to_numpy()[i]
            _p = df['Pclass'].to_numpy()[i]
            # pclassがない場合
            if _2median[_t].get(_p):
                df['Age'].to_numpy()[i] = _2median[_t][_p]
            else:
                df['Age'].to_numpy()[i] = _2median[_t].mean()
    for df in [train_df, test_df]:
        df['Age'] = pd.qcut(
            df['Age'], 10, duplicates='drop', labels=False)
    return train_df, test_df


def translate_fare(train_df, test_df):
    #######################################
    # no fare => median pclass=3, sibsp=0, parch=0
    # binning 13
    #######################################
    p2median = train_df.groupby(['Pclass', 'SibSp', 'Parch'])['Fare'].median()
    for df in [train_df, test_df]:
        for i, val in enumerate(df['Fare'].to_numpy()):
            if not np.isnan(val):
                continue

            df['Fare'].to_numpy()[i] = p2median[3][0][0]
    for df in [train_df, test_df]:
        df['Fare'] = pd.qcut(
            df['Fare'], 13, duplicates='drop', labels=False)
    return train_df, test_df


def translate_embarked(train_df, test_df):
    #######################################
    # no embarked => S
    #######################################
    for df in [train_df, test_df]:
        for i, val in enumerate(df['Embarked'].to_numpy()):
            if isinstance(val, str):
                continue
            df['Embarked'].to_numpy()[i] = 'S'
    return train_df, test_df


def translate_familysize(train_df, test_df):
    #######################################
    # sibsp + parch + 1
    # grouping 1, 2-4, 5-6, 7-11
    #######################################
    for df in [train_df, test_df]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['FamilySize'] = df['FamilySize'].replace([1], 'Alone')
        df['FamilySize'] = df['FamilySize'].replace([2, 3, 4], 'Small')
        df['FamilySize'] = df['FamilySize'].replace([5, 6], 'Medium')
        df['FamilySize'] = df['FamilySize'].replace([7, 8, 9, 10, 11], 'Large')
    return train_df, test_df


def translate_deck(train_df, test_df):
    #######################################
    # cabin[0], missing is M, T is A
    # grouping ABC, DE, FG
    #######################################
    for df in [train_df, test_df]:
        df['Deck'] = df['Cabin'].apply(
            lambda x: x[0] if isinstance(x, str) else 'M')
        df.loc[df['Deck'] == 'T', 'Deck'] = 'A'
        df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
        df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
        df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
    return train_df, test_df


def _create_nn_model():
    input_dim = 7
    activation = 'relu'
    output_dim = 2
    optimizer = 'adam'

    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation=activation))
    model.add(Dense(10, activation=activation))
    model.add(Dense(output_dim, activation="softmax"))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def create_nn_model():
    input_dim = 10

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_dim, 270)
            self.fc2 = nn.Linear(270, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = F.dropout(x, p=0.1)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.softmax(x, dim=-1)
            return x

    return Net
