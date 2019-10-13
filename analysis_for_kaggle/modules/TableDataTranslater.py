import math
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import pandas as pd
import importlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from keras.utils.np_utils import to_categorical
from IPython.display import display
from logging import getLogger

logger = getLogger('predict').getChild('TableDataTranslater')
try:
    from .CommonDataTranslater import CommonDataTranslater
except ImportError:
    logger.warn('IN FOR KERNEL SCRIPT, CommonDataTranslater import IS SKIPPED')
try:
    from .Trainer import Trainer
except ImportError:
    logger.warn('IN FOR KERNEL SCRIPT, Trainer import IS SKIPPED')


class TableDataTranslater(CommonDataTranslater):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.configs = {}

    def _categorize_ndarrays(self, model, X_train, Y_train, X_test, col_name):
        def _get_transed_data(
            model_obj, model, X_train, Y_train, target, col_name
        ):
            if model == 'label':
                model_obj = None
                feature_names = ['%s_label' % col_name]
                df = pd.DataFrame(data=X_train, columns=['x'])
                _, uniqs = pd.factorize(df['x'])
                transed = uniqs.get_indexer(target)
                transed = transed.reshape(-1, 1)
            elif model == 'count':
                model_obj = None
                feature_names = ['%s_count' % col_name]
                df = pd.DataFrame(data=X_train, columns=['x'])
                counts = df.groupby('x')['x'].count()
                transed = target
                # only test, insert 0
                transed = np.where(
                    ~np.in1d(transed, list(counts.index)), 0, transed)
                for i in counts.index:
                    transed = np.where(transed == i, counts[i], transed)
                transed = transed.reshape(-1, 1)
            elif model == 'rank':
                model_obj = None
                feature_names = ['%s_rank' % col_name]
                df = pd.DataFrame(data=X_train, columns=['x'])
                ranks = df.groupby('x')['x'].count().rank(ascending=False)
                transed = target
                # only test, insert -1
                transed = np.where(
                    ~np.in1d(transed, list(ranks.index)), -1, transed)
                for i in ranks.index:
                    transed = np.where(transed == i, ranks[i], transed)
                transed = transed.reshape(-1, 1)
            elif model == 'target':
                model_obj = None
                feature_names = ['%s_target' % col_name]
                df = pd.DataFrame(data=X_train, columns=['x'])
                df['y'] = Y_train
                means = df.groupby('x')['y'].mean()
                transed = target
                # only test, insert 0
                transed = np.where(
                    ~np.in1d(transed, list(means.index)), 0, transed)
                for i in means.index:
                    transed = np.where(transed == i, means[i], transed)
                # add Laplace Noize for not data leak
                # np.random.seed(seed=42)
                # transed += [np.random.laplace() for _ in range(len(transed))]
                transed = transed.reshape(-1, 1)
            else:
                logger.error('NOT IMPLEMENTED CATEGORIZE: %s' % model)
                raise Exception('NOT IMPLEMENTED')
            return model_obj, feature_names, transed

        output = []
        model_obj = None
        for target in [X_train, X_test]:
            model_obj, feature_names, transed = _get_transed_data(
                model_obj, model, X_train, Y_train, target, col_name)
            output.append(transed)
        return output[0], output[1], feature_names

    def translate_data_for_view(self):
        # adhoc
        trans_adhoc = self.configs['pre']['table'].get('adhoc')
        if trans_adhoc:
            if trans_adhoc['myfunc']:
                if not self.kernel:
                    myfunc = importlib.import_module(
                        'modules.myfuncs.%s' % trans_adhoc['myfunc'])
            # temp merge
            train_pred_df = pd.merge(
                self.train_df, self.pred_df, left_index=True, right_index=True)
            for method_name in trans_adhoc['methods']:
                logger.info('adhoc: %s' % method_name)
                if not self.kernel:
                    method_name = 'myfunc.%s' % method_name
                self.train_df, self.test_df = eval(
                    method_name)(train_pred_df, self.test_df)
            # temp drop
            self.train_df = train_pred_df.drop(self.pred_cols, axis=1)
            del train_pred_df
        # del
        trans_del = self.configs['pre']['table'].get('del')
        if trans_del:
            logger.info('delete: %s' % trans_del)
            self.train_df = self.train_df.drop(trans_del, axis=1)
            self.test_df = self.test_df.drop(trans_del, axis=1)
        # missing
        for column, dtype in tqdm(self.test_df.dtypes.items()):
            if column in [self.id_col]:
                continue
            if dtype == 'object':
                logger.warn('OBJECT MISSING IS NOT BE REPLACED: %s' % column)
                continue
            if (not self.train_df[column].isna().any()) \
                    and (not self.test_df[column].isna().any()):
                continue
            logger.info('replace missing with mean: %s' % column)
            column_mean = self.train_df[column].mean()
            self.train_df = self.train_df.fillna({column: column_mean})
            self.test_df = self.test_df.fillna({column: column_mean})
        # category
        trans_category = self.configs['pre']['table'].get('category')
        logger.info('categorize model: %s' % trans_category['model'])
        columns = []
        for column, dtype in tqdm(self.test_df.dtypes.items()):
            if column in [self.id_col]:
                continue
            if dtype != 'object' \
                and trans_category \
                    and column not in trans_category['target']:
                continue
            columns.append(column)
            self.train_df = self.train_df.fillna({column: 'REPLACED_NAN'})
            self.test_df = self.test_df.fillna({column: 'REPLACED_NAN'})
            if trans_category['model'] in ['onehot', 'onehot_with_test']:
                categories = self.train_df[column].unique()
                if trans_category['model'] == 'onehot_with_test':
                    logger.warn('IN DATA PREPROCESSING, USING TEST DATA')
                    categories = np.concatenate(
                        (categories, self.test_df[column].unique().tolist()),
                        axis=0)
                    categories = set(categories)
                self.train_df[column] = pd.Categorical(
                    self.train_df[column], categories=categories)
                self.test_df[column] = pd.Categorical(
                    self.test_df[column], categories=categories)
            else:
                logger.info('categorize: %s' % column)
                train_transed, test_transed, feature_names = \
                    self._categorize_ndarrays(
                        trans_category['model'],
                        self.train_df[column].values,
                        self.pred_df.values,
                        self.test_df[column].values, column)
                self.train_df = pd.merge(
                    self.train_df,
                    pd.DataFrame(train_transed, columns=feature_names),
                    left_index=True, right_index=True)
                self.train_df = self.train_df.drop([column], axis=1)
                self.test_df = pd.merge(
                    self.test_df,
                    pd.DataFrame(test_transed, columns=feature_names),
                    left_index=True, right_index=True)
                self.test_df = self.test_df.drop([column], axis=1)
        # onehot should be together
        if trans_category['model'] in ['onehot', 'onehot_with_test']:
            logger.info('categorize: %s' % columns)
            if len(columns) > 0:
                self.train_df = pd.get_dummies(
                    self.train_df, columns=columns)
                self.test_df = pd.get_dummies(
                    self.test_df, columns=columns)
        return

    def write_data_for_view(self):
        savename = self.configs['pre'].get('savename')
        if not savename:
            logger.warn('NO SAVENAME')
            return

        savename += '.csv'
        output_path = self.configs['data']['output_dir']
        self.train_df.to_csv(
            '%s/train_%s' % (output_path, savename), index=False)
        self.test_df.to_csv(
            '%s/test_%s' % (output_path, savename), index=False)
        return savename

    def create_data_for_model(self):
        # Y_train
        self.Y_train = self.pred_df.values
        # X_train
        self.X_train = sp.csr_matrix(
            self.train_df.drop(self.id_col, axis=1).values)
        # X_test
        self.test_ids = self.test_df[self.id_col].values
        self.X_test = sp.csr_matrix(
            self.test_df.drop(self.id_col, axis=1).values)
        # feature_columns
        self.feature_columns = []
        for key in self.train_df.keys():
            if key == self.id_col:
                continue
            self.feature_columns.append(key)
        return

    def _normalize_data_for_model(self):
        # x
        # scaler
        x_scaler = self.configs['pre']['table'].get('x_scaler')
        if x_scaler:
            logger.info(f'normalize x data: {x_scaler}')
            if x_scaler == 'standard':
                self.x_scaler = StandardScaler(with_mean=False)
            elif x_scaler == 'minmax':
                self.x_scaler = MinMaxScaler(with_mean=False)
            else:
                logger.error('NOT IMPLEMENTED FIT X_SCALER: %s' % x_scaler)
                raise Exception('NOT IMPLEMENTED')
            self.x_scaler.fit(self.X_train)
            self.X_train = self.x_scaler.transform(self.X_train)
            self.X_test = self.x_scaler.transform(self.X_test)
        else:
            self.x_scaler = None
        # y
        if self.configs['pre']['train_mode'] == 'reg':
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
                elif y_scaler == 'minmax':
                    self.y_scaler = MinMaxScaler()
                else:
                    logger.error('NOT IMPLEMENTED FIT Y_SCALER: %s' % y_scaler)
                    raise Exception('NOT IMPLEMENTED')
                self.y_scaler.fit(self.Y_train)
                self.Y_train = self.y_scaler.transform(self.Y_train)
            else:
                self.y_scaler = None
        return

    def _reduce_dimension_of_data_for_model(self):
        di_config = self.configs['pre']['table'].get('dimension')
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

        logger.info('reduce dimension to %s with %s' % (n, model))
        if model == 'pca':
            model_obj = PCA(n_components=n, random_state=42)
            model_obj.fit(self.X_train.toarray())
            logger.info('pca_ratio sum: %s' % sum(
                model_obj.explained_variance_ratio_))
            logger.info('pca_ratio: %s' % model_obj.explained_variance_ratio_)
        elif model == 'svd':
            model_obj = TruncatedSVD(n_components=n, random_state=42)
            model_obj.fit(self.X_train)
            logger.info('svd_ratio sum: %s' % sum(
                model_obj.explained_variance_ratio_))
            logger.info('svd_ratio: %s' % model_obj.explained_variance_ratio_)
        elif model == 'kmeans':
            model_obj = KMeans(n_clusters=n, random_state=42)
            model_obj.fit(self.X_train)
            logger.info(
                'kmeans inertia_: %s' % model_obj.inertia_)
        elif model == 'nmf':
            model_obj = NMF(n_components=n, random_state=42)
            model_obj.fit(self.X_train)
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
            model_obj.fit(self.X_train, Y_train)
        else:
            logger.error('NOT IMPLEMENTED DIMENSION MODEL: %s' % model)
            raise Exception('NOT IMPLEMENTED')

        self.X_train = model_obj.transform(self.X_train.toarray())
        self.X_test = model_obj.transform(self.X_test.toarray())
        self.feature_columns = list(map(
            lambda x: '%s_%d' % (model, x), range(n)))
        self.dimension_model = model_obj
        return

    def _extract_train_data_with_adversarial_validation(self):
        def _get_adversarial_preds(X_train, X_test, adversarial):
            # create data
            X_adv = sp.vstack((X_train, X_test), format='csr').toarray()
            Y_adv = np.concatenate(
                (np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])), axis=0)
            # fit
            trainer_obj = Trainer(**self.get_data_for_model())
            trainer_obj.configs = self.configs
            estimator = trainer_obj.calc_single_model(
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

        adversarial = self.configs['pre']['table'].get('adversarial')
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
            self.X_train = np.append(
                self.X_train, train_scores.reshape(-1, 1), axis=1)
            self.X_test = np.append(
                self.X_test, test_scores.reshape(-1, 1), axis=1)

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
        self.X_train, self.Y_train = sampler_obj.fit_resample(
            self.X_train, self.Y_train)
        logger.info('train data added %s => %s'
                    % (org_len, self.X_train.shape[0]))
        return

    def _reshape_data_for_model_for_keras(self):
        modes = self.configs['pre']['table'].get('reshape_for_keras')
        if not modes:
            return

        logger.info('reshape for keras: %s' % modes)
        # first, category
        if 'category' in modes:
            Y_train_onehot = to_categorical(self.Y_train)
            self.Y_train = Y_train_onehot
        for mode in modes:
            if mode == 'category':
                pass
            elif mode == 'lstm':
                self.X_train = self.X_train.reshape(*self.X_train.shape, 1)
                self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
                if 'category' in modes:
                    label_num = Y_train_onehot.shape[1]
                    self.X_train = np.concatenate(
                        [self.X_train] * label_num, 2)
                    self.X_test = np.concatenate(
                        [self.X_test] * label_num, 2)
            else:
                logger.error('NOT IMPLEMENTED RESHAPE FOR KERAS: %s' % mode)
                raise Exception('NOT IMPLEMENTED')
        return

    def translate_data_for_model(self):
        self._normalize_data_for_model()
        self._reduce_dimension_of_data_for_model()
        self._extract_train_data_with_adversarial_validation()
        self._extract_no_anomaly_train_data()
        self._extract_train_data_with_undersampling()
        self._add_train_data_with_oversampling()
        self._reshape_data_for_model_for_keras()
        return
