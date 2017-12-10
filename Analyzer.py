import math
import json
import configparser
import myfuncs
import numpy as np
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
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer(object):
    def __init__(self):
        self.cp = configparser.SafeConfigParser()

    def read_config_file(self, path='./config.ini'):
        self.cp.read(path)
        self.id_col = self.cp.get('data', 'id_col')
        self.pred_col = self.cp.get('data', 'pred_col')

    def read_config_text(self, text):
        self.cp.read_string(text)
        self.id_col = self.cp.get('data', 'id_col')
        self.pred_col = self.cp.get('data', 'pred_col')

    def get_base_model(self, modelname):
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

    def display_raw_data(self):
        for df in [self.train_df, self.test_df]:
            display(df.head())
            display(df.describe())

    def replace_dfs(self, dfs, target, config, *, mean=None):
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

    def categorize_dfs(self, dfs, target, config):
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

    def to_float_dfs(self, dfs, pred_col, id_col):
        output = []
        for df in dfs:
            for key in df.keys():
                if key == pred_col or key == id_col:
                    continue
                df[key] = df[key].astype(float)
            output.append(df)
        return output

    def get_raw_data(self):
        print('### DATA LIST')
        data_path = self.cp.get('data', 'path')
        print(check_output(['ls', data_path]).decode('utf8'))
        self.train_df = pd.read_csv('%s/train.csv' % data_path)
        self.test_df = pd.read_csv('%s/test.csv' % data_path)
        return self.train_df, self.test_df

    def trans_raw_data(self):
        train_df = self.train_df
        test_df = self.test_df
        cp = self.cp
        trans_adhoc = json.loads(cp.get('translate', 'adhoc'))
        trans_replace = json.loads(cp.get('translate', 'replace'))
        trans_del = json.loads(cp.get('translate', 'del'))
        trans_category = json.loads(cp.get('translate', 'category'))
        # replace
        for key, value in trans_replace.items():
            # mean
            if train_df.dtypes[key] == 'object':
                key_mean = None
            else:
                key_mean = train_df[key].mean()
            train_df, test_df = self.replace_dfs(
                [train_df, test_df], key, value, mean=key_mean)
        # adhoc
        for value in trans_adhoc:
            train_df, test_df = eval(
                'myfuncs.%s' % value)([train_df, test_df], train_df)
        # category
        for key, values in trans_category.items():
            train_df, test_df = self.categorize_dfs(
                [train_df, test_df], key, values)
        # del
        for value in trans_del:
            del train_df[value]
            del test_df[value]
        # float
        train_df, test_df = self.to_float_dfs(
            [train_df, test_df], self.pred_col, self.id_col)
        self.train_df = train_df
        self.test_df = test_df
        return self.train_df, self.test_df

    def get_fitting_data(self):
        train_df = self.train_df
        test_df = self.test_df
        # random
        if self.cp.getboolean('data', 'random'):
            train_df = train_df.iloc[np.random.permutation(len(train_df))]
        # Y_train
        self.Y_train = train_df[self.pred_col].values
        # X_train
        del train_df[self.pred_col]
        del train_df[self.id_col]
        self.X_train = train_df.values
        # X_test
        self.id_pred = test_df[self.id_col].values
        del test_df[self.id_col]
        self.X_test = test_df.values
        return self.X_train, self.Y_train, self.X_test

    def normalize_fitting_data(self):
        ss = StandardScaler()
        ss.fit(self.X_train)
        self.X_train = ss.transform(self.X_train)
        self.X_test = ss.transform(self.X_test)
        return self.X_train, self.X_test

    def extract_fitting_data_with_adversarial_validation(self):
        def _get_adversarial_score(train_num, X_train, X_test, adversarial):
            # create data
            tmp_X_train = X_train[:train_num]
            X_adv = np.concatenate((tmp_X_train, X_test), axis=0)
            target_adv = np.concatenate(
                (np.zeros(train_num), np.ones(len(X_test))), axis=0)
            # fit
            gs = GridSearchCV(
                self.get_base_model(adversarial['model']),
                adversarial['params'],
                cv=adversarial['cv'],
                scoring=adversarial['scoring'], n_jobs=-1)
            gs.fit(X_adv, target_adv)
            return gs.best_score_

        print('### DATA VALIDATION')
        X_train = self.X_train
        Y_train = self.Y_train
        adversarial = self.cp.get('data', 'adversarial')
        if adversarial:
            print('with adversarial')
            adversarial = json.loads(adversarial)
            data_unit = 100
            adv_scores = []
            for i in range(0, len(X_train) // data_unit):
                tmp_train_num = len(X_train) - i * 100
                adv_score = _get_adversarial_score(
                    tmp_train_num, X_train, self.X_test, adversarial)
                print('train num: %s, adv score: %s' % (
                    tmp_train_num, adv_score))
                adv_scores.append(adv_score)
            train_num = len(X_train) - np.argmax(adv_scores) * 100
            X_train = X_train[:train_num]
            Y_train = Y_train[:train_num]
        else:
            print('no data validation')
        print('adopted train num: %s' % len(X_train))
        self.X_train = X_train
        self.Y_train = Y_train
        return self.X_train, self.Y_train

    def calc_best_model(self):
        print('### FIT')
        base_model = self.get_base_model(self.cp.get('model', 'base'))
        scoring = self.cp.get('model', 'scoring')
        cv = self.cp.getint('model', 'cv')
        params = json.loads(self.cp.get('model', 'params'))
        gs = GridSearchCV(
            base_model, params, cv=cv, scoring=scoring, n_jobs=-1)
        gs.fit(self.X_train, self.Y_train)
        print('X train shape: %s' % str(self.X_train.shape))
        print('Y train shape: %s' % str(self.Y_train.shape))
        print('best params: %s' % gs.best_params_)
        print('best score of trained grid search: %s' % gs.best_score_)
        self.best_model = gs.best_estimator_
        print('best model: %s' % self.best_model)
        return self.best_model

    def calc_output(self, filename):
        Y_pred = self.best_model.predict(self.X_test)
        f = open('outputs/%s' % filename, 'w')
        f.write('%s,%s' % (self.id_col, self.pred_col))
        for i in range(len(self.id_pred)):
            f.write('\n')
            f.write('%s,%s' % (self.id_pred[i], Y_pred[i]))
        f.close()
        return filename

    def visualize(self):
        print('### SIMPLE VISUALIZATION')
        for key in self.train_df.keys():
            if key == self.pred_col or key == self.id_col:
                continue
            g = sns.FacetGrid(self.train_df, col=self.pred_col)
            g.map(plt.hist, key, bins=20)

if __name__ == '__main__':
    pass
