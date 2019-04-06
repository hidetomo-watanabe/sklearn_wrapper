import os
import math
import numpy as np
import pandas as pd
import importlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from IPython.display import display
from logging import getLogger

logger = getLogger('predict').getChild('DataTranslater')
try:
    from .ConfigReader import ConfigReader
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')
try:
    from .Predicter import Predicter
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, Predicter import IS SKIPPED')


class DataTranslater(ConfigReader):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
        if self.kernel:
            self.OUTPUT_PATH = '.'
        else:
            self.OUTPUT_PATH = '%s/outputs' % self.BASE_PATH
        self.configs = {}

    def display_data(self):
        if self.configs['fit']['train_mode'] == 'clf':
            logger.info('train pred counts')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                if pred_col not in self.train_df.columns:
                    logger.warn('NOT %s IN TRAIN DF' % pred_col)
                    continue
                display(self.train_df[pred_col].value_counts())
                display(self.train_df[pred_col].value_counts(normalize=True))
        elif self.configs['fit']['train_mode'] == 'reg':
            logger.info('train pred std')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                display(self.train_df[pred_col].std())
        for label, df in [('train', self.train_df), ('test', self.test_df)]:
            logger.info('%s:' % label)
            display(df.head())
            display(df.describe(include='all'))

    def _replace_missing_of_dfs(self, dfs, target, target_mean):
        replaced = False
        output = []
        for df in dfs:
            for i, val in enumerate(df[target].values):
                if math.isnan(val):
                    replaced = True
                    df[target].values[i] = target_mean
            output.append(df)
        output.insert(0, replaced)
        return output

    def _categorize_dfs(self, dfs, target):
        def _replace_nan(org):
            df = pd.DataFrame(org)
            df = df.replace({np.nan: 'DUMMY'})
            return df[0].values

        output = []
        # onehot
        train_org = dfs[0][target].values
        test_org = dfs[1][target].values
        oh_enc = OneHotEncoder(categories='auto')
        # use test data for checking category value
        oh_enc.fit(
            _replace_nan(np.concatenate([train_org, test_org])).reshape(-1, 1))
        feature_names = oh_enc.get_feature_names(input_features=[target])
        for df in dfs:
            target_org = df[target].values
            onehot = oh_enc.transform(
                _replace_nan(target_org).reshape(-1, 1)).toarray()
            for i, column in enumerate(feature_names):
                df[column] = onehot[:, i]
            del df[target]
            output.append(df)

        return output

    def _to_float_of_dfs(self, dfs, target):
        output = []
        for df in dfs:
            df[target] = df[target].astype(float)
            output.append(df)
        return output

    def get_data_for_view(self):
        output = {
            'train_df': self.train_df,
            'test_df': self.test_df,
        }
        return output

    def create_data_for_view(self):
        train_path = self.configs['data']['train_path']
        test_path = self.configs['data']['test_path']
        delim = self.configs['data'].get('delimiter')
        if delim:
            self.train_df = pd.read_csv(train_path, delimiter=delim)
            self.test_df = pd.read_csv(test_path, delimiter=delim)
        else:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
        return

    def translate_data_for_view(self):
        train_df = self.train_df
        test_df = self.test_df

        # adhoc
        trans_adhoc = self.configs['translate'].get('adhoc')
        if trans_adhoc:
            if trans_adhoc['myfunc']:
                if not self.kernel:
                    myfunc = importlib.import_module(
                        'modules.myfuncs.%s' % trans_adhoc['myfunc'])
            for method_name in trans_adhoc['methods']:
                logger.info('adhoc: %s' % method_name)
                if not self.kernel:
                    method_name = 'myfunc.%s' % method_name
                train_df, test_df = eval(
                    method_name)(train_df, test_df)
        # del
        trans_del = self.configs['translate']['del']
        if trans_del:
            for column in trans_del:
                logger.info('delete: %s' % column)
                del train_df[column]
                del test_df[column]
        # missing
        for column in test_df.columns:
            if column in [self.id_col] + self.pred_cols:
                continue
            if test_df.dtypes[column] == 'object':
                logger.warn('OBJECT MISSING IS NOT BE REPLACED: %s' % column)
                continue
            column_mean = train_df[column].mean()
            replaced, train_df, test_df = self._replace_missing_of_dfs(
                [train_df, test_df], column, column_mean)
            if replaced:
                logger.info('replace missing with mean: %s' % column)
        # category
        trans_category = self.configs['translate'].get('category')
        for column in test_df.columns:
            if column in [self.id_col] + self.pred_cols:
                continue
            if test_df.dtypes[column] != 'object' \
                and isinstance(trans_category, list) \
                    and column not in trans_category:
                continue
            logger.info('categorize: %s' % column)
            train_df, test_df = self._categorize_dfs(
                [train_df, test_df], column)
        # float
        for column in test_df.columns:
            if column in [self.id_col]:
                continue
            if self.configs['fit']['train_mode'] in ['clf'] \
                    and column in self.pred_cols:
                continue
            train_df, test_df = self._to_float_of_dfs(
                [train_df, test_df], column)
        self.train_df = train_df
        self.test_df = test_df
        return

    def write_data_for_view(self, filename='data_for_view.csv'):
        self.train_df.to_csv(
            '%s/train_%s' % (self.OUTPUT_PATH, filename), index=False)
        self.test_df.to_csv(
            '%s/test_%s' % (self.OUTPUT_PATH, filename), index=False)
        return

    def get_data_for_model(self):
        output = {
            'feature_columns': self.feature_columns,
            'test_ids': self.test_ids,
            'X_train': self.X_train,
            'Y_train': self.Y_train,
            'X_test': self.X_test,
        }
        if hasattr(self, 'y_scaler'):
            output['y_scaler'] = self.y_scaler
        return output

    def create_data_for_model(self):
        train_df = self.train_df
        test_df = self.test_df
        # Y_train
        self.Y_train = train_df[self.pred_cols].values
        # X_train
        self.X_train = train_df \
            .drop(self.id_col, axis=1).drop(self.pred_cols, axis=1).values
        # X_test
        self.test_ids = test_df[self.id_col].values
        self.X_test = test_df \
            .drop(self.id_col, axis=1).values
        # feature_columns
        self.feature_columns = []
        for key in self.train_df.keys():
            if key in self.pred_cols or key == self.id_col:
                continue
            self.feature_columns.append(key)
        return

    def normalize_data_for_model(self):
        # x
        # scaler
        logger.info('normalize x data')
        scaler_x = StandardScaler()
        scaler_x.fit(self.X_train)
        self.X_train = scaler_x.transform(self.X_train)
        self.X_test = scaler_x.transform(self.X_test)
        # y
        if self.configs['fit']['train_mode'] == 'reg':
            # pre
            y_pre = self.configs['fit'].get('y_pre')
            if y_pre:
                logger.info('translate y_train with %s' % y_pre)
                if y_pre == 'log':
                    self.Y_train = np.array(list(map(math.log, self.Y_train)))
                    self.Y_train = self.Y_train.reshape(-1, 1)
                else:
                    logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
                    raise Exception('NOT IMPLEMENTED')
            # scaler
            logger.info('normalize y data')
            y_scaler = self.configs['fit'].get('y_scaler')
            if (not y_scaler) or (y_scaler == 'standard'):
                self.y_scaler = StandardScaler()
            elif y_scaler == 'minmax':
                self.y_scaler = MinMaxScaler()
            self.y_scaler.fit(self.Y_train)
            self.Y_train = self.y_scaler.transform(self.Y_train)
        return

    def reduce_dimension_of_data_for_model(self):
        n = self.configs['translate'].get('dimension')
        if not n:
            return
        logger.info('reduce dimension with pca')
        if n == 'all':
            n = self.X_train.shape[1]
        pca_obj = PCA(n_components=n, random_state=42)
        pca_obj.fit(self.X_train)
        logger.info('pca_ratio sum: %s' % sum(
            pca_obj.explained_variance_ratio_))
        logger.info('pca_ratio: %s' % pca_obj.explained_variance_ratio_)
        self.X_train = pca_obj.transform(self.X_train)
        self.X_test = pca_obj.transform(self.X_test)
        self.feature_columns = list(map(lambda x: 'pca_%d' % x, range(n)))
        return

    def extract_train_data_with_adversarial_validation(self):
        def _get_adversarial_preds(X_train, X_test, adversarial):
            # create data
            X_adv = np.concatenate((X_train, X_test), axis=0)
            Y_adv = np.concatenate(
                (np.zeros(len(X_train)), np.ones(len(X_test))), axis=0)
            # fit
            predicter_obj = Predicter(**self.get_data_for_model())
            predicter_obj.configs = self.configs
            estimator = predicter_obj.calc_single_model(
                adversarial['scoring'], adversarial,
                X_train=X_adv, Y_train=Y_adv)
            if not hasattr(estimator, 'predict_proba'):
                logger.error(
                    'NOT PREDICT_PROBA METHOD IN ADVERSARIAL ESTIMATOR')
                raise Exception('NOT IMPLEMENTED')
            test_index = list(estimator.classes_).index(1)
            adv_train_preds = estimator.predict_proba(X_train)[:, test_index]
            adv_test_preds = estimator.predict_proba(X_test)[:, test_index]
            return adv_train_preds, adv_test_preds

        adversarial = self.configs['translate'].get('adversarial')
        if not adversarial:
            return

        logger.info('extract train data with adversarial validation')
        adv_train_preds, adv_test_preds = _get_adversarial_preds(
            self.X_train, self.X_test, adversarial)
        logger.info('adversarial train preds:')
        display(pd.DataFrame(adv_train_preds).describe(include='all'))
        logger.info('adversarial test preds:')
        display(pd.DataFrame(adv_test_preds).describe(include='all'))

        if adversarial.get('add_column'):
            logger.info('add adversarial_test_proba column to X')
            self.feature_columns.append('adversarial_test_proba')
            self.X_train = np.append(
                self.X_train, adv_train_preds.reshape(-1, 1), axis=1)
            self.X_test = np.append(
                self.X_test, adv_test_preds.reshape(-1, 1), axis=1)

        threshold = adversarial.get('threshold')
        if not threshold and int(threshold) != 0:
            threshold = 0.5
        org_len = len(self.X_train)
        self.X_train = self.X_train[adv_train_preds > threshold]
        self.Y_train = self.Y_train[adv_train_preds > threshold]
        logger.info('with threshold %s, train data reduced %s => %s'
                    % (threshold, org_len, len(self.X_train)))
        return

    def extract_train_data_with_undersampling(self):
        method = self.configs['translate'].get('undersampling')
        if not method:
            return

        logger.info('extract train data with undersampling: %s' % method)
        if method == 'random':
            sampler_obj = RandomUnderSampler(random_state=42)
        org_len = len(self.X_train)
        self.X_train, self.Y_train = sampler_obj.fit_resample(
            self.X_train, self.Y_train)
        logger.info('train data reduced %s => %s'
                    % (org_len, len(self.X_train)))
        return

    def add_train_data_with_oversampling(self):
        method = self.configs['translate'].get('oversampling')
        if not method:
            return

        logger.info('add train data with oversampling: %s' % method)
        if method == 'random':
            sampler_obj = RandomOverSampler(random_state=42)
        elif method == 'smote':
            sampler_obj = SMOTE(random_state=42)
        org_len = len(self.X_train)
        self.X_train, self.Y_train = sampler_obj.fit_resample(
            self.X_train, self.Y_train)
        logger.info('train data added %s => %s'
                    % (org_len, len(self.X_train)))
        return

    def reshape_data_for_model_for_keras(self):
        mode = self.configs['translate'].get('reshape_for_keras')
        if not mode:
            return

        logger.info('reshape for keras: %s' % mode)
        if mode == 'lstm':
            self.X_train = self.X_train.reshape(*self.X_train.shape, 1)
            self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
        elif mode == '1dcnn':
            self.X_train = self.X_train.reshape(*self.X_train.shape, 1)
            self.Y_train = self.Y_train.reshape(*self.Y_train.shape, 1)
            self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
        else:
            logger.error('NOT IMPLEMENTED RESHAPE FOR KERAS: %s' % mode)
            raise Exception('NOT IMPLEMENTED')
        return
