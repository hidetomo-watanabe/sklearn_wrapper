import os
import math
import numpy as np
import pandas as pd
import importlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import IsolationForest
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
            logger.info('train pred mean, std')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                display('mean: %f' % self.train_df[pred_col].mean())
                display('std: %f' % self.train_df[pred_col].std())
        else:
            logger.error('TRAIN MODE SHOULD BE clf OR reg')
            raise Exception('NOT IMPLEMENTED')
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

    def _categorize_dfs(self, dfs, target, model):
        def _replace_nan(org):
            df = pd.DataFrame(org)
            df = df.replace({np.nan: 'REPLACED_NAN'})
            return df[0].values

        def _get_transed_data(
            model_obj, model, fit_x_target, fit_y_target, trans_target
        ):
            if model == 'onehot':
                if not model_obj:
                    model_obj = OneHotEncoder(
                        categories='auto', handle_unknown='ignore')
                    model_obj.fit(fit_x_target.reshape(-1, 1))
                feature_names = model_obj.get_feature_names(
                    input_features=[target])
                transed = model_obj.transform(
                    trans_target.reshape(-1, 1)).toarray()
            elif model == 'label':
                model_obj = None
                feature_names = ['%s_label' % target]
                df = pd.DataFrame(data=fit_x_target, columns=['x'])
                uniqs = np.unique(df['x'].values)
                labels = pd.DataFrame(
                    data=np.arange(1, len(uniqs) + 1),
                    index=uniqs, columns=['uniq'])
                transed = trans_target
                # only test, insert -1
                transed = np.where(
                    ~np.in1d(transed, list(labels.index)), -1, transed)
                for i in labels.index:
                    transed = np.where(
                        transed == i, labels['uniq'][i], transed)
                transed = transed.reshape(-1, 1)
            elif model == 'count':
                model_obj = None
                feature_names = ['%s_count' % target]
                df = pd.DataFrame(data=fit_x_target, columns=['x'])
                counts = df.groupby('x')['x'].count()
                transed = trans_target
                # only test, insert 1
                transed = np.where(
                    ~np.in1d(transed, list(counts.index)), 1, transed)
                for i in counts.index:
                    transed = np.where(transed == i, counts[i], transed)
                transed = transed.reshape(-1, 1)
            elif model == 'rank':
                model_obj = None
                feature_names = ['%s_rank' % target]
                df = pd.DataFrame(data=fit_x_target, columns=['x'])
                ranks = df.groupby('x')['x'].count().rank(ascending=False)
                transed = trans_target
                # only test, insert -1
                transed = np.where(
                    ~np.in1d(transed, list(ranks.index)), -1, transed)
                for i in ranks.index:
                    transed = np.where(transed == i, ranks[i], transed)
                transed = transed.reshape(-1, 1)
            elif model == 'target':
                model_obj = None
                feature_names = ['%s_target' % target]
                df = pd.DataFrame(data=fit_x_target, columns=['x'])
                df['y'] = fit_y_target
                means = df.groupby('x')['y'].mean()
                transed = trans_target
                # only test, insert 0
                transed = np.where(
                    ~np.in1d(transed, list(means.index)), 0, transed)
                for i in means.index:
                    transed = np.where(transed == i, means[i], transed)
                transed = transed.reshape(-1, 1)
            else:
                logger.error('NOT IMPLEMENTED CATEGORIZE: %s' % model)
                raise Exception('NOT IMPLEMENTED')
            return model_obj, feature_names, transed

        output = []
        fit_x_target = _replace_nan(dfs[0][target].values)
        fit_y_target = _replace_nan(dfs[0][self.pred_cols].values)
        model_obj = None
        for df in dfs:
            trans_target = _replace_nan(df[target].values)
            model_obj, feature_names, transed = _get_transed_data(
                model_obj, model, fit_x_target, fit_y_target, trans_target)
            for i, column in enumerate(feature_names):
                df[column] = transed[:, i]
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
        trans_del = self.configs['translate'].get('del')
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
        logger.info('categorize model: %s' % trans_category['model'])
        for column in test_df.columns:
            if column in [self.id_col] + self.pred_cols:
                continue
            if test_df.dtypes[column] != 'object' \
                and trans_category \
                    and column not in trans_category['target']:
                continue
            logger.info('categorize: %s' % column)
            train_df, test_df = self._categorize_dfs(
                [train_df, test_df], column, trans_category['model'])
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

    def write_data_for_view(self):
        savename = self.configs['translate'].get('savename')
        if not savename:
            logger.warn('NO SAVENAME')
            return

        savename += '.csv'
        self.train_df.to_csv(
            '%s/train_%s' % (self.OUTPUT_PATH, savename), index=False)
        self.test_df.to_csv(
            '%s/test_%s' % (self.OUTPUT_PATH, savename), index=False)
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
        di_config = self.configs['translate'].get('dimension')
        if not di_config:
            return

        n = di_config['n']
        model = di_config['model']
        logger.info('reduce dimension to %s with %s' % (n, model))
        if n == 'all':
            n = self.X_train.shape[1]

        if model == 'pca':
            model_obj = PCA(n_components=n, random_state=42)
            model_obj.fit(self.X_train)
            logger.info('pca_ratio sum: %s' % sum(
                model_obj.explained_variance_ratio_))
            logger.info('pca_ratio: %s' % model_obj.explained_variance_ratio_)
        elif model == 'rfe':
            # for warning
            Y_train = self.Y_train
            if len(Y_train.shape) > 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()
            model_obj = RFE(
                n_features_to_select=n,
                estimator=XGBClassifier(random_state=42, n_jobs=-1))
            model_obj.fit(self.X_train, Y_train)
        else:
            logger.error('NOT IMPLEMENTED DIMENSION MODEL: %s' % model)
            raise Exception('NOT IMPLEMENTED')

        self.X_train = model_obj.transform(self.X_train)
        self.X_test = model_obj.transform(self.X_test)
        self.feature_columns = list(map(
            lambda x: '%s_%d' % (model, x), range(n)))
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
            if hasattr(estimator, 'classes_'):
                test_index = list(estimator.classes_).index(1)
            else:
                logger.warn('CLASSES_ NOT IN ESTIMATOR')
                test_index = 1
            adv_train_preds = estimator.predict_proba(X_train)[:, test_index]
            adv_test_preds = estimator.predict_proba(X_test)[:, test_index]
            return adv_train_preds, adv_test_preds

        adversarial = self.configs['translate'].get('adversarial')
        if not adversarial:
            return

        logger.info('extract train data with adversarial validation')
        logger.warn('IN DATA PREPROCESSING, USING TEST DATA')
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

    def extract_no_anomaly_train_data(self):
        if not self.configs['translate'].get('no_anomaly'):
            return

        logger.info('extract no anomaly train data')
        isf = IsolationForest(random_state=42, n_jobs=-1)
        preds = isf.fit_predict(self.X_train, self.Y_train)
        org_len = len(self.X_train)
        self.X_train = self.X_train[preds == 1]
        self.Y_train = self.Y_train[preds == 1]
        logger.info('train data reduced %s => %s'
                    % (org_len, len(self.X_train)))
        return

    def extract_train_data_with_undersampling(self):
        method = self.configs['translate'].get('undersampling')
        if not method:
            return

        logger.info('extract train data with undersampling: %s' % method)
        if method == 'random':
            sampler_obj = RandomUnderSampler(random_state=42)
        else:
            logger.error('NOT IMPLEMENTED UNDERSAMPLING METHOD: %s' % method)
            raise Exception('NOT IMPLEMENTED')
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
        else:
            logger.error('NOT IMPLEMENTED OVERSAMPLING METHOD: %s' % method)
            raise Exception('NOT IMPLEMENTED')
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
