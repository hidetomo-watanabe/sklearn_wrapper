import math
import numpy as np
import pandas as pd
import importlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from logging import getLogger

logger = getLogger('predict').getChild('DataTranslater')
try:
    from .ConfigReader import ConfigReader
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')


class DataTranslater(ConfigReader):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.configs = {}

    def display_data(self):
        if self.configs['fit']['train_mode'] == 'clf':
            logger.info('train pred counts')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
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

    def get_raw_data(self):
        output = {
            'train_df': self.train_df,
            'test_df': self.test_df,
        }
        return output

    def create_raw_data(self):
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

    def translate_raw_data(self):
        train_df = self.train_df
        test_df = self.test_df
        trans_adhoc = self.configs['translate']['adhoc']
        # adhoc
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
        for column in self.configs['translate']['del']:
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
        for column in test_df.columns:
            if column in [self.id_col] + self.pred_cols:
                continue
            if test_df.dtypes[column] != 'object' \
                    and column not in self.configs['translate']['category']:
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

    def get_data_for_model(self):
        output = {
            'feature_columns': self.feature_columns,
            'test_ids': self.test_ids,
            'X_train': self.X_train,
            'Y_train': self.Y_train,
            'X_test': self.X_test,
        }
        if hasattr(self, 'scaler_y'):
            output['scaler_y'] = self.scaler_y
        return output

    def create_data_for_model(self):
        train_df = self.train_df
        test_df = self.test_df
        # Y_train
        if len(self.pred_cols) == 1:
            self.Y_train = train_df[self.pred_cols[0]].values
        else:
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
        # ss
        scaler_x = StandardScaler()
        scaler_x.fit(self.X_train)
        self.X_train = scaler_x.transform(self.X_train)
        self.X_test = scaler_x.transform(self.X_test)
        # y
        if self.configs['fit']['train_mode'] == 'reg':
            # other
            y_pre = self.configs['fit']['y_pre']
            if y_pre:
                logger.info('translate y_train with %s' % y_pre)
                if y_pre == 'log':
                    self.Y_train = np.array(list(map(math.log, self.Y_train)))
                else:
                    logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
                    raise Exception('NOT IMPLEMENTED')
            # ss
            self.scaler_y = StandardScaler()
            self.Y_train = self.Y_train.reshape(-1, 1)
            self.scaler_y.fit(self.Y_train)
            self.Y_train = self.scaler_y.transform(self.Y_train).reshape(-1, )
        return

    def reduce_dimension_of_data_for_model(self):
        n = self.configs['translate']['dimension']
        if not n:
            return
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
