import math
from keras.models import Sequential
from keras.layers.core import Dense
from torch import nn
import torch.nn.functional as F


def translate_honorific_title(train_df, test_df):
    #######################################
    # name => honorific title
    #######################################
    train_values = []
    test_values = []
    for i, val in enumerate(train_df['Name'].to_numpy()):
        title = val.split(',')[1].split('.')[0].strip()
        if title in ['Mr', 'Mrs', 'Miss', 'Master']:
            train_values.append(title)
        else:
            train_values.append('Other')
    for i, val in enumerate(test_df['Name'].to_numpy()):
        title = val.split(',')[1].split('.')[0].strip()
        if title in ['Mr', 'Mrs', 'Miss', 'Master']:
            test_values.append(title)
        else:
            test_values.append('Other')
    train_df['HonorificTitle'] = train_values
    test_df['HonorificTitle'] = test_values
    return train_df, test_df


def translate_age(train_df, test_df):
    #######################################
    # no age => mean grouped Mr, Mrs, Miss
    #######################################
    # get honorific title => age mean
    t2m = {}
    tmp = train_df.groupby('HonorificTitle')['Age']
    for key in tmp.indices.keys():
        t2m[key] = tmp.mean()[key]
    # age range
    for df in [train_df, test_df]:
        for i, val in enumerate(df['Age'].to_numpy()):
            if math.isnan(val):
                df['Age'].to_numpy()[i] = \
                    t2m[df['HonorificTitle'].to_numpy()[i]]
    return train_df, test_df


def translate_fare(train_df, test_df):
    #######################################
    # no fare => mean grouped by pclass
    #######################################
    for df in [train_df, test_df]:
        for i, val in enumerate(df['Fare'].to_numpy()):
            if math.isnan(val):
                df['Fare'].to_numpy()[i] = \
                    train_df.groupby('Pclass')['Fare'].mean()[
                        df['Pclass'].to_numpy()[i]]
    return train_df, test_df


def translate_familystatus(train_df, test_df):
    #######################################
    # sibsp + parch == 0 => no family(0)
    # survive vs no survive in same familyname
    # => more survive(1) or less survive(2)
    # delete sibsp, parch
    # categorize after
    #######################################
    # get family name
    train_df['FamilyName'] = [''] * len(train_df['Name'].to_numpy())
    for i, val in enumerate(train_df['Name'].to_numpy()):
        train_df['FamilyName'].to_numpy()[i] = val.split(',')[0]
    # get family name => family status
    n2s = {}
    tmp = train_df.groupby('FamilyName')['Survived']
    for key in tmp.indices.keys():
        survived_num = tmp.sum()[key]
        no_survived_num = tmp.count()[key] - survived_num
        if survived_num > no_survived_num:
            n2s[key] = 1
        else:
            n2s[key] = 2
    # main
    for df in [train_df, test_df]:
        df['FamilyStatus'] = [0] * len(df['Name'].to_numpy())
        for i, val in enumerate(df['Name'].to_numpy()):
            family_name = val.split(',')[0]
            # no family
            if df['SibSp'].to_numpy()[i] + df['Parch'].to_numpy()[i] == 0:
                continue
            # any family
            if family_name in n2s:
                df['FamilyStatus'].to_numpy()[i] = n2s[family_name]
        # del sibsp, parch
        del df['SibSp']
        del df['Parch']
    # del family name
    del train_df['FamilyName']
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
    input_dim = 7

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
