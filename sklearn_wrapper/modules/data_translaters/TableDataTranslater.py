import importlib
from logging import getLogger

from IPython.display import display

from boruta import BorutaPy

from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder

import numpy as np

import pandas as pd

import scipy.sparse as sp
from scipy.stats import ks_2samp

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from xgboost import XGBClassifier

logger = getLogger('predict').getChild('TableDataTranslater')
if 'BaseDataTranslater' not in globals():
    from .BaseDataTranslater import BaseDataTranslater
if 'SingleTrainer' not in globals():
    from ..trainers.SingleTrainer import SingleTrainer
if 'Trainer' not in globals():
    from ..trainers.Trainer import Trainer


class TableDataTranslater(BaseDataTranslater):
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

    def _encode_category_with_target(self, columns):
        model_obj = TargetEncoder(cols=columns)
        # cvをずらせないので、train_cvを採用
        cv, _ = Trainer.get_cvs_from_json(self.configs['fit'].get('cv'))
        logger.info(f'cv: {cv}')
        indexes = cv.split(self.train_df, self.pred_df)

        # train
        train_encoded = []
        for train_index, pred_index in indexes:
            model_obj.fit(
                self.train_df.loc[train_index][columns],
                self.pred_df.loc[train_index])
            _train_encoded = model_obj.transform(
                self.train_df.loc[pred_index][columns])
            train_encoded.append(_train_encoded)
        train_encoded = pd.concat(train_encoded)
        train_encoded.sort_index(inplace=True)

        # test
        # train全てでfit
        model_obj.fit(self.train_df[columns], self.pred_df)
        test_encoded = model_obj.transform(self.test_df[columns])
        self.target_encoding_model = model_obj

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

    def _encode_category_with_ndarray(self, model, columns):
        train_encoded = []
        test_encoded = []
        feature_names = []
        for column in tqdm(columns):
            self.train_df.fillna({column: 'REPLACED_NAN'}, inplace=True)
            self.test_df.fillna({column: 'REPLACED_NAN'}, inplace=True)
            _train_encoded = self._encode_ndarray(
                model, self.train_df[column].to_numpy(),
                self.pred_df.to_numpy(), self.train_df[column].to_numpy())
            _test_encoded = self._encode_ndarray(
                model, self.train_df[column].to_numpy(),
                self.pred_df.to_numpy(), self.test_df[column].to_numpy())
            train_encoded.append(_train_encoded)
            test_encoded.append(_test_encoded)
            feature_names.append(f'{column}_{model}')
        train_encoded = pd.DataFrame(
            np.concatenate(train_encoded, axis=1), columns=feature_names)
        test_encoded = pd.DataFrame(
            np.concatenate(test_encoded, axis=1), columns=feature_names)
        return train_encoded, test_encoded

    def _encode_category_single(self, model, columns):
        if model in [
            'onehot', 'onehot_with_test', 'label', 'label_with_test', 'target'
        ]:
            if model == 'target':
                train_encoded, test_encoded = \
                    self._encode_category_with_target(columns)
            else:
                if model in ['onehot', 'onehot_with_test']:
                    model_obj = OneHotEncoder(cols=columns, use_cat_names=True)
                elif model in ['label', 'label_with_test']:
                    model_obj = OrdinalEncoder(cols=columns)
                if model in ['onehot_with_test', 'label_with_test']:
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
                rename_mapping[column] = f'{column}_{model}'
            train_encoded.rename(columns=rename_mapping, inplace=True)
            test_encoded.rename(columns=rename_mapping, inplace=True)
        else:
            train_encoded, test_encoded = \
                self._encode_category_with_ndarray(model, columns)

        # merge
        self.train_df = pd.merge(
            self.train_df, train_encoded,
            left_index=True, right_index=True)
        self.test_df = pd.merge(
            self.test_df, test_encoded,
            left_index=True, right_index=True)
        return

    def _encode_category(self):
        trans_category = self.configs['pre']['table'].get('category_encoding')
        if not trans_category:
            return

        # default columns
        option_columns = []
        for option in trans_category['options']:
            option_columns.extend(option['columns'])
        option_columns = list(set(option_columns))
        default_columns = []
        for column, dtype in self.test_df.dtypes.items():
            if column in [self.id_col]:
                continue
            if dtype != 'object':
                continue
            if column in option_columns:
                continue
            default_columns.append(column)

        # encode
        drop_columns = []
        trans_category['default']['columns'] = default_columns
        for config in trans_category['options'] + [trans_category['default']]:
            drop_columns.extend(config['columns'])
            logger.info('encoding model: %s' % config['model'])
            logger.info('encode category: %s' % config['columns'])
            if 'with_test' in config['model']:
                logger.warning('IN DATA PREPROCESSING, USING TEST DATA')
            self._encode_category_single(config['model'], config['columns'])
        drop_columns = list(set(drop_columns))

        # drop
        self.train_df.drop(drop_columns, axis=1, inplace=True)
        self.test_df.drop(drop_columns, axis=1, inplace=True)
        return

    def _calc_base_train_data(self):
        self.Y_train = self.pred_df.to_numpy()
        self.train_ids = self.train_df[self.id_col].to_numpy()
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

    def _to_float32(self):
        if self.X_train.dtype != 'object':
            self.X_train = self.X_train.astype(np.float32)
            self.X_test = self.X_test.astype(np.float32)
        if self.configs['pre']['train_mode'] == 'reg':
            self.Y_train = self.Y_train.astype(np.float32)
        return

    def _impute_missing_value(self):
        imputation = self.configs['pre']['table'].get('missing_imputation')
        if not imputation:
            return

        model_obj = SimpleImputer(missing_values=np.nan, strategy=imputation)
        model_obj.fit(self.X_train)
        self.X_train = model_obj.transform(self.X_train)
        self.X_test = model_obj.transform(self.X_test)
        return

    def _select_feature(self):
        selection = self.configs['pre']['table'].get('feature_selection')
        if not selection:
            return

        n = selection['n']
        model = selection['model']
        if model == 'boruta':
            selector = BorutaPy(
                estimator=RandomForestClassifier(
                    class_weight='balanced', max_depth=5, n_jobs=-1),
                n_estimators=n, verbose=2, random_state=42)
        elif model == 'rfe':
            selector = RFE(
                estimator=XGBClassifier(random_state=42, n_jobs=-1),
                n_features_to_select=n)
        else:
            logger.error(
                'NOT IMPLEMENTED FEATURE SELECTION: %s' % model)
            raise Exception('NOT IMPLEMENTED')

        selector.fit(self.X_train, self.ravel_like(self.Y_train))
        features = selector.support_
        logger.info(
            f'select feature {self.X_train.shape[1]}'
            f' to {len(features[features])}')
        self.X_train = self.X_train[:, features]
        self.X_test = self.X_test[:, features]
        self.feature_columns = list(np.array(self.feature_columns)[features])
        return

    def _extract_with_ks_validation(self):
        ks = self.configs['pre']['table'].get('ks_validation')
        if not ks:
            return

        logger.info('extract columns with Kolmogorov-Smirnov validation')
        _indexes = []
        for i, col in enumerate(self.feature_columns):
            p_val = ks_2samp(self.X_train[:, i], self.X_test[:, i])[1]
            if p_val < 0.05:
                logger.info(
                    'Kolmogorov-Smirnov not same distriburion: %s'
                    % self.feature_columns[i])
            else:
                _indexes.append(i)
        self.X_train = self.X_train[:, _indexes]
        self.X_test = self.X_test[:, _indexes]
        self.feature_columns = list(np.array(self.feature_columns)[_indexes])
        return

    def _extract_with_adversarial_validation(self):
        def _get_adversarial_preds(X_train, X_test, adversarial):
            if adversarial['model_config'].get('cv_select') != 'train_all':
                logger.error(
                    'ONLY IMPLEMENTED ADVERSARIAL VALIDATION WITH TRAIN ALL')
                raise Exception('NOT IMPLEMENTED')

            # create data
            X_adv = np.vstack((X_train, X_test))
            Y_adv = np.concatenate(
                (np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])),
                axis=0)
            # fit
            single_trainer_obj = SingleTrainer(
                X_train=self.X_train, Y_train=self.Y_train, X_test=self.X_test,
                feature_columns=self.feature_columns, configs=self.configs)
            _, estimator = single_trainer_obj.calc_single_estimator(
                adversarial['model_config'], X_train=X_adv, Y_train=Y_adv)
            if not hasattr(estimator, 'predict_proba'):
                logger.error(
                    'NOT PREDICT_PROBA METHOD IN ADVERSARIAL ESTIMATOR')
                raise Exception('NOT IMPLEMENTED')
            if hasattr(estimator, 'classes_'):
                test_index = list(estimator.classes_).index(1)
            else:
                logger.warning('CLASSES_ NOT IN ESTIMATOR')
                test_index = 1
            # predict
            auc = roc_auc_score(
                Y_adv, estimator.predict_proba(X_adv)[:, test_index])
            logger.info(f'auc: {auc}')
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
            self.X_train = np.hstack(
                (self.X_train,
                 np.array(adv_train_preds.reshape(-1, 1))))
            self.X_test = np.hstack(
                (self.X_test,
                 np.array(adv_test_preds.reshape(-1, 1))))

        threshold = adversarial.get('threshold', 0.5)
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
            contamination=contamination, random_state=42, n_jobs=-1)
        preds = isf.fit_predict(self.X_train, self.Y_train)
        train_scores = isf.decision_function(self.X_train)
        test_scores = isf.decision_function(self.X_test)

        if no_anomaly.get('add_column'):
            logger.info('add no_anomaly_score column to X')
            self.feature_columns.append('no_anomaly_score')
            self.X_train = np.hstack(
                (self.X_train,
                 np.array(train_scores.reshape(-1, 1))))
            self.X_test = np.hstack(
                (self.X_test,
                 np.array(test_scores.reshape(-1, 1))))

        org_len = self.X_train.shape[0]
        self.X_train = self.X_train[preds == 1]
        self.Y_train = self.Y_train[preds == 1]
        logger.info('train data reduced %s => %s'
                    % (org_len, self.X_train.shape[0]))
        return

    def _reshape_x_for_keras(self):
        mode = self.configs['pre']['table'].get('reshape_for_keras')
        if not mode:
            return

        logger.info('reshape x for keras: %s' % mode)
        if mode == 'lstm':
            self.X_train = self.X_train.reshape(*self.X_train.shape, 1)
            self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
            label_num = len(np.unique(self.Y_train))
            self.X_train = np.concatenate([self.X_train] * label_num, axis=2)
            self.X_test = np.concatenate([self.X_test] * label_num, axis=2)
        else:
            logger.error('NOT IMPLEMENTED RESHAPE FOR KERAS: %s' % mode)
            raise Exception('NOT IMPLEMENTED')
        return

    def _to_sparse(self):
        sparse = self.configs['pre']['table'].get('sparse')
        if not sparse:
            return

        if self.X_train.dtype == 'object':
            return

        if 'g_nb' not in [
            _c['model'] for _c
            in self.configs['fit']['single_model_configs']
        ]:
            logger.info('set x to sparse')
            self.X_train = sp.csr_matrix(self.X_train)
            self.X_test = sp.csr_matrix(self.X_test)
        return

    def calc_train_data(self):
        # df
        self._calc_raw_data()
        self._translate_adhoc_df()
        self._delete_columns()
        self._encode_category()
        # ndarray
        self._calc_base_train_data()
        self._translate_adhoc_ndarray()
        self._to_float32()
        self._translate_y_pre()
        # validation
        self._impute_missing_value()
        self._select_feature()
        self._extract_with_ks_validation()
        self._extract_with_adversarial_validation()
        self._extract_with_no_anomaly_validation()
        # format
        self._reshape_x_for_keras()
        self._to_sparse()
        return
