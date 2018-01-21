import math
import json
import pickle
import numpy as np
import pandas as pd
import importlib
from subprocess import check_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer(object):
    def __init__(self):
        self.configs = {}

    def read_config_file(self, path='./config.json'):
        with open(path, 'r') as f:
            self.configs = json.loads(f.read())
        self.id_col = self.configs['data']['id_col']
        self.pred_col = self.configs['data']['pred_col']

    def read_config_text(self, text):
        self.configs = json.loads(text)
        self.id_col = self.configs['data']['id_col']
        self.pred_col = self.configs['data']['pred_col']

    def _get_base_model(self, modelname):
        if modelname == 'log_reg':
            return LogisticRegression()
        elif modelname == 'linear_reg':
            return LinearRegression()
        elif modelname == 'svc':
            return SVC()
        elif modelname == 'svr':
            return SVR()
        elif modelname == 'l_svc':
            return LinearSVC()
        elif modelname == 'l_svr':
            return LinearSVR()
        elif modelname == 'rf_clf':
            return RandomForestClassifier()
        elif modelname == 'rf_reg':
            return RandomForestRegressor()
        elif modelname == 'gbdt_clf':
            return GradientBoostingClassifier()
        elif modelname == 'gbdt_reg':
            return GradientBoostingRegressor()
        elif modelname == 'knn_clf':
            return KNeighborsClassifier()
        elif modelname == 'knn_reg':
            return KNeighborsRegressor()
        elif modelname == 'g_nb':
            return GaussianNB()
        elif modelname == 'preceptron':
            return Perceptron()
        elif modelname == 'sgd_clf':
            return SGDClassifier()
        elif modelname == 'sgd_reg':
            return SGDRegressor()
        elif modelname == 'dt_clf':
            return DecisionTreeClassifier()
        elif modelname == 'dt_reg':
            return DecisionTreeRegressor()
        elif modelname == 'xgb_clf':
            return XGBClassifier()
        elif modelname == 'xgb_reg':
            return XGBRegressor()

    def display_data(self):
        print('### DATA LIST')
        for label, df in [('train', self.train_df), ('test', self.test_df)]:
            print('%s:' % label)
            display(df.head())
            display(df.describe())

    def _replace_missing_of_dfs(self, dfs, target, target_mean):
        replaced = False
        output = [replaced]
        for df in dfs:
            for i, val in enumerate(df[target].values):
                if math.isnan(val):
                    replaced = True
                    df[target].values[i] = target_mean
            output.append(df)
        return output

    def _categorize_dfs(self, dfs, target):
        output = []
        all_vals = []
        for df in dfs:
            all_vals.extend(list(set(df[target].values)))
        for df in dfs:
            for val in all_vals:
                df['%s_%s' % (target, val)] = [0] * len(df[target].values)
            for i, val_tmp in enumerate(df[target].values):
                for val in all_vals:
                    if val_tmp == val:
                        df['%s_%s' % (target, val)].values[i] = 1
            del df[target]
            output.append(df)
        return output

    def _reduce_dimension_of_dfs(self, dfs, train_df):
        def _drop_id_pred_cols(df):
            if self.pred_col in df.columns:
                return df.drop(self.id_col, axis=1).drop(self.pred_col, axis=1)
            else:
                return df.drop(self.id_col, axis=1)

        output = []
        n = self.configs['translate']['dimension']
        if not n:
            return dfs
        if n == 'all':
            n = len(_drop_id_pred_cols(train_df).columns)
        pca_obj = PCA(n_components=n)
        pca_obj.fit(_drop_id_pred_cols(train_df))
        print('pca_ratio: %s' % pca_obj.explained_variance_ratio_)
        for df_org in dfs:
            df = pd.DataFrame(
                pca_obj.transform(_drop_id_pred_cols(df_org).values))
            df[self.id_col] = df_org[self.id_col]
            if self.pred_col in df_org.columns:
                df[self.pred_col] = df_org[self.pred_col].values
            output.append(df)
        return output

    def _to_float_of_dfs(self, dfs, target):
        output = []
        for df in dfs:
            df[target] = df[target].astype(float)
            output.append(df)
        return output

    def get_raw_data(self):
        train_path = self.configs['data']['train_path']
        test_path = self.configs['data']['test_path']
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        return self.train_df, self.test_df

    def trans_raw_data(self):
        train_df = self.train_df
        test_df = self.test_df
        trans_adhoc = self.configs['translate']['adhoc']
        # adhoc
        if trans_adhoc['myfunc']:
            myfunc = importlib.import_module(
                'myfuncs.%s' % trans_adhoc['myfunc'])
        for method_name in trans_adhoc['methods']:
            print('adhoc: %s' % method_name)
            train_df, test_df = eval(
                'myfunc.%s' % method_name)([train_df, test_df], train_df)
        # del
        for column in self.configs['translate']['del']:
            print('del: %s' % column)
            del train_df[column]
            del test_df[column]
        # missing
        for column in test_df.columns:
            if column in [self.id_col, self.pred_col]:
                continue
            if test_df.dtypes[column] == 'object':
                continue
            column_mean = train_df[column].mean()
            replaced, train_df, test_df = self._replace_missing_of_dfs(
                [train_df, test_df], column, column_mean)
            if replaced:
                print('missing: %s' % column)
        # category
        for column in test_df.columns:
            if column in [self.id_col, self.pred_col]:
                continue
            if (
                test_df.dtypes[column] != 'object' and
                column not in self.configs['translate']['category']
            ):
                continue
            print('category: %s' % column)
            train_df, test_df = self._categorize_dfs(
                [train_df, test_df], column)
        # dimension
        train_df, test_df = self._reduce_dimension_of_dfs(
            [train_df, test_df], train_df)
        # float
        for column in test_df.columns:
            if column in [self.pred_col]:
                continue
            if (
                self.configs['fit']['mode'] == 'clf' and
                column in [self.id_col]
            ):
                continue
            train_df, test_df = self._to_float_of_dfs(
                [train_df, test_df], column)
        self.train_df = train_df
        self.test_df = test_df
        return self.train_df, self.test_df

    def get_fitting_data(self):
        train_df = self.train_df
        test_df = self.test_df
        # random
        if self.configs['data']['random']:
            print('randomize train data')
            train_df = train_df.iloc[np.random.permutation(len(train_df))]
        # Y_train
        self.Y_train = train_df[self.pred_col].values
        # X_train
        self.X_train = train_df \
            .drop(self.id_col, axis=1).drop(self.pred_col, axis=1).values
        # X_test
        self.id_pred = test_df[self.id_col].values
        self.X_test = test_df \
            .drop(self.id_col, axis=1).values
        return self.X_train, self.Y_train, self.X_test

    def normalize_fitting_data(self):
        # x
        ss_x = StandardScaler()
        ss_x.fit(self.X_train)
        self.X_train = ss_x.transform(self.X_train)
        self.X_test = ss_x.transform(self.X_test)
        # y
        if self.configs['fit']['mode'] == 'reg':
            self.ss_y = StandardScaler()
            self.Y_train = self.Y_train.reshape(-1, 1)
            self.ss_y.fit(self.Y_train)
            self.Y_train = self.ss_y.transform(self.Y_train).reshape(-1, )
        return self.X_train, self.Y_train, self.X_test

    def is_ok_with_adversarial_validation(self):
        def _get_adversarial_preds(X_train, X_test, adversarial):
            # create data
            tmp_X_train = X_train[:len(X_test)]
            X_adv = np.concatenate((tmp_X_train, X_test), axis=0)
            target_adv = np.concatenate(
                (np.zeros(len(X_test)), np.ones(len(X_test))), axis=0)
            # fit
            gs = GridSearchCV(
                self._get_base_model(adversarial['model']),
                adversarial['params'],
                cv=adversarial['cv'],
                scoring=adversarial['scoring'],
                n_jobs=adversarial['n_jobs'])
            gs.fit(X_adv, target_adv)
            est = gs.best_estimator_
            return est.predict(tmp_X_train), est.predict(X_test)

        def _is_ok_pred_nums(tr0, tr1, te0, te1):
            if tr0 == 0 and te0 == 0:
                return False
            if tr1 == 0 and te1 == 0:
                return False
            if tr1 == 0 and te0 == 0:
                return False
            return True

        print('### DATA VALIDATION')
        X_train = self.X_train
        adversarial = self.configs['data']['adversarial']
        if adversarial:
            print('with adversarial')
            adv_pred_train, adv_pred_test = _get_adversarial_preds(
                X_train, self.X_test, adversarial)
            adv_pred_train_num_0 = len(np.where(adv_pred_train == 0)[0])
            adv_pred_train_num_1 = len(np.where(adv_pred_train == 1)[0])
            adv_pred_test_num_0 = len(np.where(adv_pred_test == 0)[0])
            adv_pred_test_num_1 = len(np.where(adv_pred_test == 1)[0])
            print('pred train num 0: %s' % adv_pred_train_num_0)
            print('pred train num 1: %s' % adv_pred_train_num_1)
            print('pred test num 0: %s' % adv_pred_test_num_0)
            print('pred test num 1: %s' % adv_pred_test_num_1)
            if not _is_ok_pred_nums(
                adv_pred_train_num_0,
                adv_pred_train_num_1,
                adv_pred_test_num_0,
                adv_pred_test_num_1,
            ):
                raise Exception(
                    '[ERROR] TRAIN AND TEST MAY BE HAVE DIFFERENT FEATURES')
        else:
            print('[WARN] NO DATA VALIDATION')
            return True

    def calc_best_model(self, filename):
        print('### FIT')
        estimators = []
        for i, modelname in enumerate(
            self.configs['fit']['models']
        ):
            base_model = self._get_base_model(modelname)
            scoring = self.configs['fit']['scoring']
            cv = self.configs['fit']['cv']
            n_jobs = self.configs['fit']['n_jobs']
            params = self.configs['fit']['params'][i]
            gs = GridSearchCV(
                base_model, params, cv=cv, scoring=scoring, n_jobs=n_jobs)
            gs.fit(self.X_train, self.Y_train)
            print('modelname: %s' % modelname)
            print('  X train shape: %s' % str(self.X_train.shape))
            print('  Y train shape: %s' % str(self.Y_train.shape))
            print('  best params: %s' % gs.best_params_)
            print('  best score of trained grid search: %s' % gs.best_score_)
            estimators.append((modelname, gs.best_estimator_))
        # classification
        if self.configs['fit']['mode'] == 'clf':
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                weights=[1] * len(estimators),
                voting='hard', n_jobs=n_jobs)
        # regression
        elif self.configs['fit']['mode'] == 'reg':
            if len(self.configs['fit']['models']) != 1:
                raise Exception(
                    '[ERROR] WHEN MODE IS REG, MODEL SHOULD BE 1')
            self.ensemble_model = BaggingRegressor(
                base_estimator=estimators[0][1],
                n_jobs=n_jobs)
        self.ensemble_model = self.ensemble_model.fit(
            self.X_train, self.Y_train)
        print('ensemble model: %s' % self.ensemble_model)
        with open('outputs/%s' % filename, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        return self.ensemble_model

    def calc_output(self, filename):
        Y_pred = self.ensemble_model.predict(self.X_test)
        if self.configs['fit']['mode'] == 'reg':
            Y_pred = self.ss_y.inverse_transform(Y_pred)
        with open('outputs/%s' % filename, 'w') as f:
            f.write('%s,%s' % (self.id_col, self.pred_col))
            for i in range(len(self.id_pred)):
                f.write('\n')
                f.write('%s,%s' % (self.id_pred[i], Y_pred[i]))
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
