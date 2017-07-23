import math
import json
import configparser
import myfuncs
# import numpy as np
import pandas as pd
from IPython.display import display
from subprocess import check_output
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
"""
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
"""


def get_base_model(modelname):
    if modelname == 'log_reg':
        return LogisticRegression()
    elif modelname == 'svc':
        return SVC()
    elif modelname == 'l_svc':
        return LinearSVC()
    elif modelname == 'rf_clf':
        return RandomForestClassifier()
    elif modelname == 'gbdt_clf':
        return GradientBoostingClassifier()
    elif modelname == 'knn_clf':
        return KNeighborsClassifier()
    elif modelname == 'g_nb':
        return GaussianNB()
    elif modelname == 'preceptron':
        return Perceptron()
    elif modelname == 'sgd_clf':
        return SGDClassifier()
    elif modelname == 'dt_clf':
        return DecisionTreeClassifier()


def display_dfs(dfs):
    for df in dfs:
        display(df.head())
        display(df.describe())


def translate_dfs(dfs, target, config, *, mean=None):
    """
        config is dict
    """
    output = []
    for df in dfs:
        for i, value1 in enumerate(df[target].values):
            for key, value2 in config.items():
                # mean
                if value2 == 'mean':
                    value2 = mean
                # translate
                if key == 'isnan':
                    if math.isnan(value1):
                        df[target].values[i] = value2
                else:
                    if value1 == key:
                        df[target].values[i] = value2
        output.append(df)
    return output


def categorize_dfs(dfs, target, config):
    """
        config is list
    """
    output = []
    for df in dfs:
        for value2 in config:
            df['%s_%s' % (target, value2)] = [0] * len(df[target].values)
        for i, value1 in enumerate(df[target].values):
            for value2 in config:
                if value1 == value2:
                    df['%s_%s' % (target, value2)].values[i] = 1
        del df[target]
        output.append(df)
    return output

if __name__ == '__main__':
    # definition
    cp = configparser.SafeConfigParser()
    cp.read('./config.ini')
    # data
    data_path = cp.get('data', 'path')
    pred_col = cp.get('data', 'pred_col')
    id_col = cp.get('data', 'id_col')
    train_num = cp.get('data', 'train_num')
    if train_num:
        train_num = int(train_num)
    else:
        train_num = None
    # model
    base_model = get_base_model(cp.get('model', 'base'))
    scoring = cp.get('model', 'scoring')
    params = json.loads(cp.get('model', 'params'))
    # traslate
    trans_adhoc = json.loads(cp.get('translate', 'adhoc'))
    trans_replace = json.loads(cp.get('translate', 'replace'))
    trans_del = json.loads(cp.get('translate', 'del'))
    trans_category = json.loads(cp.get('translate', 'category'))

    # data list
    print('### DATA LIST')
    print(check_output(['ls', data_path]).decode('utf8'))
    train_df = pd.read_csv('%strain.csv' % data_path)
    test_df = pd.read_csv('%stest.csv' % data_path)

    # init overview
    print('### INIT OVERVIEW')
    display_dfs([train_df, test_df])

    # translate df
    print('### DATA TRANSLATION')
    # replace
    for key, value in trans_replace.items():
        # mean
        if train_df.dtypes[key] == 'object':
            key_mean = None
        else:
            key_mean = train_df[key].mean()
        translate_dfs([train_df, test_df], key, value, mean=key_mean)
        print('##### %s' % key)
        display_dfs([train_df[key], test_df[key]])
    # adhoc
    for value in trans_adhoc:
        eval('myfuncs.%s' % value)([train_df, test_df], train_df)
    # category
    for key, values in trans_category.items():
        categorize_dfs([train_df, test_df], key, values)
        print('##### %s' % key)
        for val in values:
            display_dfs([
                train_df['%s_%s' % (key, val)],
                test_df['%s_%s' % (key, val)]])
    # del
    for value in trans_del:
        del train_df[value]
        del test_df[value]

    # translation overview
    print('### TRANSLATION OVERVIEW')
    display_dfs([train_df, test_df])

    """
    # simple visualization
    print('### SIMPLE VISUALIZATION')
    for value in train_df.columns:
        if value == pred_col or value == id_col:
            continue
        if train_df.dtypes[value] == 'object':
            continue
        print('##### value: %s' % value)
        g = sns.FacetGrid(train_df, col=pred_col)
        g.map(plt.hist, value, bins=20)
        print(train_df.groupby(pred_col)[value].mean())
    """

    # create ndarray
    Y_train = train_df[pred_col].values
    id_pred = test_df[id_col].values
    del train_df[pred_col]
    del train_df[id_col]
    del test_df[id_col]
    X_train = train_df.values
    X_test = test_df.values
    display(X_train)
    display(Y_train)
    display(X_test)

    # translate ndarray to standard
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    # fit and predict
    gs = GridSearchCV(base_model, params, scoring=scoring, n_jobs=-1)
    if train_num:
        gs.fit(X_train[:train_num], Y_train[:train_num])
    else:
        gs.fit(X_train, Y_train)
    display(gs.best_params_)
    display(gs.best_score_)
    best_model = gs.best_estimator_
    display(best_model)
    Y_pred = best_model.predict(X_test)

    # create output
    f = open('outputs/tmp.csv', 'w')
    f.write('%s,%s' % (id_col, pred_col))
    for i in range(len(id_pred)):
        f.write('\n')
        f.write('%s,%s' % (id_pred[i], Y_pred[i]))
    f.close()
