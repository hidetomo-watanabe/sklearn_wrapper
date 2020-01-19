import math
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import pandas as pd
import importlib
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from IPython.display import display
from logging import getLogger

logger = getLogger('predict').getChild('TableDataTranslater')
try:
    from .CommonDataTranslater import CommonDataTranslater
except ImportError:
    logger.warning(
        'IN FOR KERNEL SCRIPT, CommonDataTranslater import IS SKIPPED')
try:
    from .Trainer import Trainer
except ImportError:
    logger.warning('IN FOR KERNEL SCRIPT, Trainer import IS SKIPPED')


class TableDataTranslater(CommonDataTranslater):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.configs = {}

    def _translate_adhoc_df(self):
        trans_adhoc_df = self.configs['pre']['table'].get('adhoc_df')
        if not trans_adhoc_df:
            return

        if not self.kernel:
            myfunc = importlib.import_module(
                'modules.myfuncs.%s' % trans_adhoc_df['myfunc'])
        # temp merge
        train_pred_df = pd.merge(
            self.train_df, self.pred_df, left_index=True, right_index=True)
        for method_name in trans_adhoc_df['methods']:
            logger.info('adhoc_df: %s' % method_name)
            if not self.kernel:
                method_name = 'myfunc.%s' % method_name
            train_pred_df, self.test_df = eval(
                method_name)(train_pred_df, self.test_df)
        # temp drop
        self.train_df = train_pred_df.drop(self.pred_cols, axis=1)
        del train_pred_df
        return

    def _delete_columns(self):
        trans_del = self.configs['pre']['table'].get('del')
        if not trans_del:
            return

        logger.info('delete: %s' % trans_del)
        self.train_df.drop(trans_del, axis=1, inplace=True)
        self.test_df.drop(trans_del, axis=1, inplace=True)
        return

    def _fill_missing_value_with_mean(self):
        trans_missing = self.configs['pre']['table'].get('missing')
        if not trans_missing:
            return

        for column, dtype in tqdm(self.test_df.dtypes.items()):
            if column in [self.id_col]:
                continue
            if (not self.train_df[column].isna().any()) \
                    and (not self.test_df[column].isna().any()):
                continue
            if dtype == 'object':
                logger.warning(
                    'OBJECT MISSING IS NOT BE REPLACED: %s' % column)
                continue
            logger.info('replace missing with mean: %s' % column)
            column_mean = self.train_df[column].mean()
            self.train_df.fillna({column: column_mean}, inplace=True)
            self.test_df.fillna({column: column_mean}, inplace=True)
        return

    def _categorize_ndarrays(self, model, X_train, Y_train, X_test, col_name):
        def _get_transed_data(
            model, X_train, Y_train, target, col_name
        ):
            feature_names = [f'{col_name}_{model}']
            df = pd.DataFrame(data=X_train, columns=['x'])
            if model == 'label':
                _, uniqs = pd.factorize(df['x'])
                transed = uniqs.get_indexer(target)
            elif model in ['count', 'freq', 'rank', 'target']:
                transed = target
                if model == 'count':
                    mapping = df.groupby('x')['x'].count()
                    only_test = 0
                elif model == 'freq':
                    mapping = df.groupby('x')['x'].count() / len(df)
                    only_test = 0
                elif model == 'rank':
                    mapping = df.groupby('x')['x'].count().rank(
                        ascending=False)
                    only_test = -1
                elif model == 'target':
                    df['y'] = Y_train
                    mapping = df.groupby('x')['y'].mean()
                    only_test = 0
                for i in mapping.index:
                    transed = np.where(transed == i, mapping[i], transed)
                transed = np.where(
                    ~np.in1d(transed, list(mapping.index)), only_test, transed)
            else:
                logger.error('NOT IMPLEMENTED CATEGORIZE: %s' % model)
                raise Exception('NOT IMPLEMENTED')
            transed = transed.reshape(-1, 1)
            return feature_names, transed

        output = []
        for target in [X_train, X_test]:
            feature_names, transed = _get_transed_data(
                model, X_train, Y_train, target, col_name)
            output.append(transed)
        return output[0], output[1], feature_names

    def _categorize(self):
        trans_category = self.configs['pre']['table'].get('category')
        if not trans_category:
            return

        logger.info('categorize model: %s' % trans_category['model'])
        columns = []
        for column, dtype in tqdm(self.test_df.dtypes.items()):
            if column in [self.id_col]:
                continue
            if dtype != 'object' and column not in trans_category['target']:
                continue
            columns.append(column)
            self.train_df.fillna({column: 'REPLACED_NAN'}, inplace=True)
            self.test_df.fillna({column: 'REPLACED_NAN'}, inplace=True)
            # onehot
            if trans_category['model'] in ['onehot', 'onehot_with_test']:
                categories = self.train_df[column].unique()
                if trans_category['model'] == 'onehot_with_test':
                    logger.warning('IN DATA PREPROCESSING, USING TEST DATA')
                    categories = np.concatenate(
                        (categories, self.test_df[column].unique().tolist()),
                        axis=0)
                    categories = set(categories)
                # pre onehot
                self.train_df[column] = pd.Categorical(
                    self.train_df[column], categories=categories)
                self.test_df[column] = pd.Categorical(
                    self.test_df[column], categories=categories)
            # not onehot
            else:
                logger.info('categorize: %s' % column)
                train_transed, test_transed, feature_names = \
                    self._categorize_ndarrays(
                        trans_category['model'],
                        self.train_df[column].to_numpy(),
                        self.pred_df.to_numpy(),
                        self.test_df[column].to_numpy(), column)
                self.train_df = pd.merge(
                    self.train_df,
                    pd.DataFrame(train_transed, columns=feature_names),
                    left_index=True, right_index=True)
                self.train_df.drop([column], axis=1, inplace=True)
                self.test_df = pd.merge(
                    self.test_df,
                    pd.DataFrame(test_transed, columns=feature_names),
                    left_index=True, right_index=True)
                self.test_df.drop([column], axis=1, inplace=True)
        # onehot should be together
        if trans_category['model'] in ['onehot', 'onehot_with_test']:
            logger.info('categorize: %s' % columns)
            if len(columns) > 0:
                self.train_df = pd.get_dummies(
                    self.train_df, columns=columns)
                self.test_df = pd.get_dummies(
                    self.test_df, columns=columns)
        return

    def _calc_base_train_data(self):
        self.Y_train = self.pred_df.to_numpy()
        self.X_train = self.train_df.drop(
            self.id_col, axis=1).to_numpy().astype('float32')
        self.test_ids = self.test_df[self.id_col].to_numpy()
        self.X_test = self.test_df.drop(
            self.id_col, axis=1).to_numpy().astype('float32')
        self.feature_columns = []
        for key in self.train_df.keys():
            if key == self.id_col:
                continue
            self.feature_columns.append(key)
        return

    def _translate_adhoc_ndarray(self):
        trans_adhoc_ndarray = \
            self.configs['pre']['table'].get('adhoc_ndarray')
        if not trans_adhoc_ndarray:
            return

        if not self.kernel:
            myfunc = importlib.import_module(
                'modules.myfuncs.%s' % trans_adhoc_ndarray['myfunc'])
        for method_name in trans_adhoc_ndarray['methods']:
            logger.info('adhoc_ndarray: %s' % method_name)
            if not self.kernel:
                method_name = 'myfunc.%s' % method_name
            self.X_train, self.X_test, self.feature_columns = eval(
                method_name)(self.X_train, self.X_test, self.feature_columns)
        return

    def _to_sparse(self):
        self.X_train = sp.csr_matrix(self.X_train)
        self.X_test = sp.csr_matrix(self.X_test)
        return

    def _normalize_x_data_for_model(self):
        # scaler
        x_scaler = self.configs['pre']['table'].get('x_scaler')
        if x_scaler:
            logger.info(f'normalize x data: {x_scaler}')
            if x_scaler == 'standard':
                self.x_scaler = StandardScaler(with_mean=False)
            elif x_scaler == 'maxabs':
                self.x_scaler = MaxAbsScaler()
            else:
                logger.error('NOT IMPLEMENTED FIT X_SCALER: %s' % x_scaler)
                raise Exception('NOT IMPLEMENTED')
            self.x_scaler.fit(self.X_train)
            self.X_train = self.x_scaler.transform(self.X_train)
            self.X_test = self.x_scaler.transform(self.X_test)
        else:
            self.x_scaler = None
        return

    def _normalize_y_data_for_model(self):
        if self.configs['pre']['train_mode'] != 'reg':
            return
        # pre
        y_pre = self.configs['pre'].get('y_pre')
        if y_pre:
            logger.info('translate y_train with %s' % y_pre)
            if y_pre == 'log':
                self.Y_train = np.array(list(map(math.log, self.Y_train)))
                self.Y_train = self.Y_train.reshape(-1, 1)
            else:
                logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
                raise Exception('NOT IMPLEMENTED')
        # scaler
        y_scaler = self.configs['pre'].get('y_scaler')
        if y_scaler:
            logger.info(f'normalize y data: {y_scaler}')
            if y_scaler == 'standard':
                self.y_scaler = StandardScaler()
            elif y_scaler == 'maxabs':
                self.y_scaler = MaxAbsScaler()
            else:
                logger.error('NOT IMPLEMENTED FIT Y_SCALER: %s' % y_scaler)
                raise Exception('NOT IMPLEMENTED')
            self.y_scaler.fit(self.Y_train)
            self.Y_train = self.y_scaler.transform(self.Y_train)
        else:
            self.y_scaler = None
        return

    def _reduce_dimension_of_data_for_model(self):
        di_config = self.configs['pre']['table'].get('dimension_reduction')
        if not di_config:
            return

        n = di_config['n']
        model = di_config['model']
        if n == 'all':
            n = self.X_train.shape[1]

        if model == 'ks':
            logger.info('reduce dimension with %s' % model)
            for i, col in enumerate(self.feature_columns):
                p_val = ks_2samp(self.X_train[:, i], self.X_test[:, i])[1]
                if p_val < 0.1:
                    logger.info(
                        'Kolmogorov-Smirnov not same distriburion: %s'
                        % self.feature_columns[i])
                    self.X_train = np.delete(self.X_train, i, 1)
                    self.X_test = np.delete(self.X_test, i, 1)
                    del self.feature_columns[i]
            return

        logger.info(
            'reduce dimension %s to %s with %s'
            % (self.X_train.shape[1], n, model))
        X_train = self.X_train
        X_test = self.X_test
        if model == 'pca':
            X_train = X_train.toarray()
            X_test = X_test.toarray()

        if model == 'pca':
            model_obj = PCA(n_components=n, random_state=42)
            model_obj.fit(X_train)
            ratios = model_obj.explained_variance_ratio_
            logger.info('pca_ratio sum: %s' % sum(ratios))
            if len(np.unique(ratios)) != len(ratios):
                logger.warning(
                    'PCA VARIANCE RATIO IS NOT UNIQUE, SO NOT REPRODUCIBLE')
            # logger.info('pca_ratio: %s' % ratios)
        elif model == 'svd':
            model_obj = TruncatedSVD(n_components=n, random_state=42)
            model_obj.fit(X_train)
            logger.info('svd_ratio sum: %s' % sum(
                model_obj.explained_variance_ratio_))
            logger.info('svd_ratio: %s' % model_obj.explained_variance_ratio_)
        elif model == 'kmeans':
            model_obj = KMeans(n_clusters=n, random_state=42)
            model_obj.fit(X_train)
            logger.info(
                'kmeans inertia_: %s' % model_obj.inertia_)
        elif model == 'nmf':
            model_obj = NMF(n_components=n, random_state=42)
            model_obj.fit(X_train)
            logger.info(
                'nmf reconstruction_err_: %s' % model_obj.reconstruction_err_)
        elif model == 'rfe':
            # for warning
            Y_train = self.Y_train
            if Y_train.ndim > 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()
            model_obj = RFE(
                n_features_to_select=n,
                estimator=XGBClassifier(random_state=42, n_jobs=-1))
            model_obj.fit(X_train, Y_train)
        else:
            logger.error(
                'NOT IMPLEMENTED DIMENSION REDUCTION MODEL: %s' % model)
            raise Exception('NOT IMPLEMENTED')

        self.X_train = model_obj.transform(X_train)
        self.X_test = model_obj.transform(X_test)
        self.feature_columns = list(map(
            lambda x: '%s_%d' % (model, x), range(n)))
        self.dimension_reduction_model = model_obj
        return

    def _extract_train_data_with_adversarial_validation(self):
        def _get_adversarial_preds(X_train, X_test, adversarial):
            if adversarial['model_config'].get('cv_select') == 'all_folds':
                logger.error(
                    'NOT IMPLEMENTED ADVERSARIAL VALIDATION WITH ALL FOLDS')
                raise Exception('NOT IMPLEMENTED')

            # create data
            X_adv = sp.vstack((X_train, X_test), format='csr')
            Y_adv = sp.csr_matrix(np.concatenate(
                (np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])),
                axis=0))
            # fit
            trainer_obj = Trainer(**self.get_train_data())
            trainer_obj.configs = self.configs
            _, estimators = trainer_obj.calc_single_estimators(
                adversarial['scoring'], adversarial['model_config'],
                X_train=X_adv, Y_train=Y_adv)
            estimator = estimators[0]
            if not hasattr(estimator, 'predict_proba'):
                logger.error(
                    'NOT PREDICT_PROBA METHOD IN ADVERSARIAL ESTIMATOR')
                raise Exception('NOT IMPLEMENTED')
            if hasattr(estimator, 'classes_'):
                test_index = list(estimator.classes_).index(1)
            else:
                logger.warning('CLASSES_ NOT IN ESTIMATOR')
                test_index = 1
            adv_train_preds = estimator.predict_proba(X_train)[:, test_index]
            adv_test_preds = estimator.predict_proba(X_test)[:, test_index]
            return adv_train_preds, adv_test_preds

        adversarial = self.configs['pre']['table'].get('adversarial')
        if not adversarial:
            return

        logger.info('extract train data with adversarial validation')
        logger.warning('IN DATA PREPROCESSING, USING TEST DATA')
        adv_train_preds, adv_test_preds = _get_adversarial_preds(
            self.X_train, self.X_test, adversarial)
        logger.info('adversarial train preds:')
        display(pd.DataFrame(adv_train_preds).describe(include='all'))
        logger.info('adversarial test preds:')
        display(pd.DataFrame(adv_test_preds).describe(include='all'))

        if adversarial.get('add_column'):
            logger.info('add adversarial_test_proba column to X')
            self.feature_columns.append('adversarial_test_proba')
            self.X_train = sp.hstack(
                (self.X_train,
                 sp.csr_matrix(adv_train_preds.reshape(-1, 1))),
                format='csr')
            self.X_test = sp.hstack(
                (self.X_test,
                 sp.csr_matrix(adv_test_preds.reshape(-1, 1))),
                format='csr')

        threshold = adversarial.get('threshold')
        if not threshold and int(threshold) != 0:
            threshold = 0.5
        org_len = self.X_train.shape[0]
        self.X_train = self.X_train[adv_train_preds > threshold]
        self.Y_train = self.Y_train[adv_train_preds > threshold]
        logger.info('with threshold %s, train data reduced %s => %s'
                    % (threshold, org_len, self.X_train.shape[0]))
        return

    def _extract_no_anomaly_train_data(self):
        no_anomaly = self.configs['pre']['table'].get('no_anomaly')
        if not no_anomaly:
            return

        logger.info('extract no anomaly train data')
        contamination = no_anomaly.get('contamination')
        if not contamination and int(contamination) != 0:
            contamination = 'auto'
        isf = IsolationForest(
            contamination=contamination,
            behaviour='new', random_state=42, n_jobs=-1)
        preds = isf.fit_predict(self.X_train, self.Y_train)
        train_scores = isf.decision_function(self.X_train)
        test_scores = isf.decision_function(self.X_test)

        if no_anomaly.get('add_column'):
            logger.info('add no_anomaly_score column to X')
            self.feature_columns.append('no_anomaly_score')
            self.X_train = sp.hstack(
                (self.X_train,
                 sp.csr_matrix(train_scores.reshape(-1, 1))),
                format='csr')
            self.X_test = sp.hstack(
                (self.X_test,
                 sp.csr_matrix(test_scores.reshape(-1, 1))),
                format='csr')

        org_len = self.X_train.shape[0]
        self.X_train = self.X_train[preds == 1]
        self.Y_train = self.Y_train[preds == 1]
        logger.info('train data reduced %s => %s'
                    % (org_len, self.X_train.shape[0]))
        return

    def _extract_train_data_with_undersampling(self):
        method = self.configs['pre']['table'].get('undersampling')
        if not method:
            return

        logger.info('extract train data with undersampling: %s' % method)
        if method == 'random':
            sampler_obj = RandomUnderSampler(random_state=42)
        else:
            logger.error('NOT IMPLEMENTED UNDERSAMPLING METHOD: %s' % method)
            raise Exception('NOT IMPLEMENTED')
        org_len = self.X_train.shape[0]
        self.X_train, self.Y_train = sampler_obj.fit_resample(
            self.X_train, self.Y_train)
        logger.info('train data reduced %s => %s'
                    % (org_len, self.X_train.shape[0]))
        return

    def _add_train_data_with_oversampling(self):
        method = self.configs['pre']['table'].get('oversampling')
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
        org_len = self.X_train.shape[0]
        Y_train = self.Y_train
        if Y_train.ndim > 1 and Y_train.shape[1] == 1:
            Y_train = Y_train.ravel()
        self.X_train, self.Y_train = sampler_obj.fit_resample(
            self.X_train, Y_train)
        logger.info('train data added %s => %s'
                    % (org_len, self.X_train.shape[0]))
        return

    def _reshape_data_for_model_for_keras(self):
        mode = self.configs['pre']['table'].get('reshape_for_keras')
        if not mode:
            return

        logger.info('reshape for keras: %s' % mode)
        if mode == 'lstm':
            self.X_train = self.X_train.reshape(*self.X_train.shape, 1)
            self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
            label_num = len(np.unique(self.Y_train))
            self.X_train = np.concatenate([self.X_train] * label_num, 2)
            self.X_test = np.concatenate([self.X_test] * label_num, 2)
        else:
            logger.error('NOT IMPLEMENTED RESHAPE FOR KERAS: %s' % mode)
            raise Exception('NOT IMPLEMENTED')
        return

    def calc_train_data(self):
        # df
        self._calc_raw_data()
        self._translate_adhoc_df()
        self._delete_columns()
        self._fill_missing_value_with_mean()
        self._categorize()
        # ndarray
        self._calc_base_train_data()
        self._translate_adhoc_ndarray()
        self._to_sparse()
        self._normalize_x_data_for_model()
        self._normalize_y_data_for_model()
        self._reduce_dimension_of_data_for_model()
        self._extract_train_data_with_adversarial_validation()
        self._extract_no_anomaly_train_data()
        self._extract_train_data_with_undersampling()
        self._add_train_data_with_oversampling()
        self._reshape_data_for_model_for_keras()
        return
