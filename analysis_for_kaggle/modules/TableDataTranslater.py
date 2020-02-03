import math
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import pandas as pd
import importlib
from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from IPython.display import display
from logging import getLogger

logger = getLogger('predict').getChild('TableDataTranslater')
if 'CommonMethodWrapper' not in globals():
    from .CommonMethodWrapper import CommonMethodWrapper
if 'BaseDataTranslater' not in globals():
    from .BaseDataTranslater import BaseDataTranslater
if 'Trainer' not in globals():
    from .Trainer import Trainer


class TableDataTranslater(CommonMethodWrapper, BaseDataTranslater):
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
        trans_del = self.configs['pre']['table'].get('deletion')
        if not trans_del:
            return

        logger.info('delete: %s' % trans_del)
        self.train_df.drop(trans_del, axis=1, inplace=True)
        self.test_df.drop(trans_del, axis=1, inplace=True)
        return

    def _impute_missing_value_with_mean(self):
        trans_missing = self.configs['pre']['table'].get('missing_imputation')
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
                    'OBJECT MISSING IS NOT BE IMPUTED: %s' % column)
                continue
            logger.info('impute missing with mean: %s' % column)
            column_mean = self.train_df[column].mean()
            self.train_df.fillna({column: column_mean}, inplace=True)
            self.test_df.fillna({column: column_mean}, inplace=True)
        return

    def _encode_category_with_target(self, columns):
        model_obj = TargetEncoder(cols=columns)
        cv = Trainer.get_cv_from_json(
            self.configs['fit'].get('cv'),
            self.configs['fit']['train_mode'])
        indexes = cv.split(self.train_df, self.pred_df)
        train_encoded = []
        for train_index, pred_index in indexes:
            model_obj.fit(
                self.train_df.loc[train_index][columns],
                self.pred_df.loc[train_index])
            _train_encoded = model_obj.transform(
                self.train_df.loc[pred_index][columns])
            train_encoded.append(_train_encoded)
        train_encoded = pd.concat(train_encoded, ignore_index=True)
        model_obj.fit(self.train_df[columns], self.pred_df)
        test_encoded = model_obj.transform(self.test_df[columns])
        return train_encoded, test_encoded

    def _encode_ndarray(self, model, X_train, Y_train, X_data):
        df = pd.DataFrame(data=X_train, columns=['x'])
        encoded = X_data
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
        else:
            logger.error('NOT IMPLEMENTED CATEGORY ENCODING: %s' % model)
            raise Exception('NOT IMPLEMENTED')

        for i in mapping.index:
            encoded = np.where(encoded == i, mapping[i], encoded)
        encoded = np.where(
            ~np.in1d(encoded, list(mapping.index)), only_test, encoded)
        encoded = encoded.reshape(-1, 1)
        return encoded

    def _encode_category_with_ndarray(self, columns):
        train_encoded = []
        test_encoded = []
        feature_names = []
        for column in tqdm(columns):
            self.train_df.fillna({column: 'REPLACED_NAN'}, inplace=True)
            self.test_df.fillna({column: 'REPLACED_NAN'}, inplace=True)
            _train_encoded = self._encode_ndarray(
                trans_category['model'], self.train_df[column].to_numpy(),
                self.pred_df.to_numpy(), self.train_df[column].to_numpy())
            _test_encoded = self._encode_ndarray(
                trans_category['model'], self.train_df[column].to_numpy(),
                self.pred_df.to_numpy(), self.test_df[column].to_numpy())
            train_encoded.append(_train_encoded)
            test_encoded.append(_test_encoded)
            feature_names.append(f'{column}_{trans_category["model"]}')
        train_encoded = pd.DataFrame(
            np.concatenate(train_encoded, axis=1), columns=feature_names)
        test_encoded = pd.DataFrame(
            np.concatenate(test_encoded, axis=1), columns=feature_names)
        return train_encoded, test_encoded

    def _encode_category(self):
        trans_category = self.configs['pre']['table'].get('category_encoding')
        if not trans_category:
            return

        # columns
        columns = []
        for column, dtype in self.test_df.dtypes.items():
            if column in [self.id_col]:
                continue
            if dtype != 'object' and column not in trans_category['option']:
                continue
            columns.append(column)
        if len(columns) == 0:
            return
        logger.info('encode category: %s' % columns)

        # encode
        logger.info('encoding model: %s' % trans_category['model'])
        if 'with_test' in trans_category['model']:
            logger.warning('IN DATA PREPROCESSING, USING TEST DATA')
        if trans_category['model'] in [
            'onehot', 'onehot_with_test', 'label', 'label_with_test', 'target'
        ]:
            if trans_category['model'] == 'target':
                train_encoded, test_encoded = \
                    self._encode_category_with_target(columns)
            else:
                if trans_category['model'] in ['onehot', 'onehot_with_test']:
                    model_obj = OneHotEncoder(cols=columns, use_cat_names=True)
                elif trans_category['model'] in ['label', 'label_with_test']:
                    model_obj = OrdinalEncoder(cols=columns)
                if trans_category['model'] in [
                    'onehot_with_test', 'label_with_test'
                ]:
                    model_obj.fit(
                        pd.concat(
                            [self.train_df[columns], self.test_df[columns]],
                            ignore_index=True
                        ))
                else:
                    model_obj.fit(self.train_df[columns], self.pred_df)
                train_encoded = model_obj.transform(
                    self.train_df[columns], self.pred_df)
                test_encoded = model_obj.transform(self.test_df[columns])
            # rename
            rename_mapping = {}
            for column in columns:
                rename_mapping[column] = f'{column}_{trans_category["model"]}'
            train_encoded.rename(columns=rename_mapping, inplace=True)
            test_encoded.rename(columns=rename_mapping, inplace=True)
        else:
            train_encoded, test_encoded = \
                self._encode_category_with_ndarray(columns)

        # merge
        self.train_df = pd.merge(
            self.train_df, train_encoded,
            left_index=True, right_index=True)
        self.test_df = pd.merge(
            self.test_df, test_encoded,
            left_index=True, right_index=True)

        # drop
        self.train_df.drop(columns, axis=1, inplace=True)
        self.test_df.drop(columns, axis=1, inplace=True)
        return

    def _calc_base_train_data(self):
        self.Y_train = self.pred_df.to_numpy()
        self.X_train = self.train_df.drop(
            self.id_col, axis=1).to_numpy()
        self.test_ids = self.test_df[self.id_col].to_numpy()
        self.X_test = self.test_df.drop(
            self.id_col, axis=1).to_numpy()
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
        self.X_train = sp.csr_matrix(self.X_train.astype('float32'))
        self.X_test = sp.csr_matrix(self.X_test.astype('float32'))
        return

    def _normalize_x(self):
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

    def _normalize_y(self):
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

    def _reduce_dimension(self):
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
            Y_train = self.ravel_like(self.Y_train)
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

    def _select_feature(self):
        selection = self.configs['pre']['table'].get('feature_selection')
        if not selection:
            return

        rf = RandomForestClassifier(
            n_jobs=-1, class_weight='balanced', max_depth=5)
        selector = BorutaPy(
            rf, n_estimators='auto', verbose=2, random_state=42)
        Y_train = self.ravel_like(self.Y_train)
        selector.fit(self.X_train.toarray(), Y_train)
        features = selector.support_
        logger.info(
            f'select feature {self.X_train.shape[1]}'
            f' to {len(features[features])}')
        self.X_train = self.X_train[:, features]
        self.X_test = self.X_test[:, features]
        self.feature_columns = list(np.array(self.feature_columns)[features])
        return

    def _extract_with_adversarial_validation(self):
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

        adversarial = \
            self.configs['pre']['table'].get('adversarial_validation')
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

    def _extract_with_no_anomaly_validation(self):
        no_anomaly = self.configs['pre']['table'].get('no_anomaly_validation')
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

    def _sample_with_under(self):
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

    def _sample_with_over(self):
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
        Y_train = self.ravel_like(self.Y_train)
        self.X_train, self.Y_train = sampler_obj.fit_resample(
            self.X_train, Y_train)
        logger.info('train data added %s => %s'
                    % (org_len, self.X_train.shape[0]))
        return

    def _reshape_for_keras(self):
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
        self._impute_missing_value_with_mean()
        self._encode_category()
        # ndarray
        self._calc_base_train_data()
        self._translate_adhoc_ndarray()
        self._to_sparse()
        self._normalize_x()
        self._normalize_y()
        self._reduce_dimension()
        self._select_feature()
        self._extract_with_adversarial_validation()
        self._extract_with_no_anomaly_validation()
        self._sample_with_under()
        self._sample_with_over()
        self._reshape_for_keras()
        return
