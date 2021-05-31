import importlib
from logging import getLogger

import numpy as np

import pandas as pd


logger = getLogger('predict').getChild('Outputer')
if 'ConfigReader' not in globals():
    from .ConfigReader import ConfigReader
if 'LikeWrapper' not in globals():
    from .commons.LikeWrapper import LikeWrapper
if 'MyKerasClassifier' not in globals():
    from .commons.MyKeras import MyKerasClassifier


class Outputer(ConfigReader, LikeWrapper):
    def __init__(
        self,
        feature_columns, train_ids, test_ids,
        X_train, Y_train, X_test,
        train_cv, val_cv, scorer, classes, single_estimators, estimator,
        kernel=False
    ):
        self.feature_columns = feature_columns
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.classes = classes
        self.estimator = estimator
        self.kernel = kernel
        self.configs = {}

    def get_predict_data(self):
        output = {
            'Y_pred': self.Y_pred,
            'Y_pred_proba': self.Y_pred_proba,
            'Y_pred_df': self.Y_pred_df,
            'Y_pred_proba_df': self.Y_pred_proba_df,
        }
        return output

    @classmethod
    def _trans_xy_for_predict(self, estimator, X_train, Y_train, X_target):
        Y_train = self.ravel_like(Y_train)
        return X_train, Y_train, X_target

    @classmethod
    def predict_like(
        self, train_mode, estimator, X_train, Y_train, X_target
    ):
        X_train, Y_train, X_target = \
            self._trans_xy_for_predict(estimator, X_train, Y_train, X_target)

        Y_pred_proba = None
        # clf
        if train_mode == 'clf':
            # keras
            if estimator.__class__ in [MyKerasClassifier] and \
                    Y_train.ndim == 2 and Y_train.shape[1] > 1:
                Y_pred = estimator.predict_proba(X_target)
            else:
                Y_pred = estimator.predict(X_target)
                if hasattr(estimator, 'predict_proba'):
                    Y_pred_proba = estimator.predict_proba(
                        X_target)
        # reg
        elif train_mode == 'reg':
            Y_pred = estimator.predict(X_target)
        else:
            logger.error('TRAIN MODE SHOULD BE clf OR reg')
            raise Exception('NOT IMPLEMENTED')
        return Y_pred, Y_pred_proba

    def _inverse_translate_y_pre(self):
        if self.configs['pre']['train_mode'] != 'reg':
            return
        y_pre = self.configs['pre'].get('y_pre')
        if not y_pre:
            return

        logger.info('inverse translate y_pred with %s' % y_pre)
        if y_pre == 'log':
            self.Y_pred = np.array(list(map(np.exp, self.Y_pred)))
        else:
            logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
            raise Exception('NOT IMPLEMENTED')
        return

    def _calc_base_predict_df(self):
        self.Y_pred_df = pd.merge(
            pd.DataFrame(data=self.test_ids, columns=[self.id_col]),
            pd.DataFrame(data=self.Y_pred, columns=self.pred_cols),
            left_index=True, right_index=True)

        if self.Y_pred_proba is None:
            self.Y_pred_proba_df = None
            return

        if self.Y_pred_proba.shape[1] == self.classes.shape[0]:
            self.Y_pred_proba_df = pd.DataFrame(
                data=self.test_ids, columns=[self.id_col])
            if len(self.pred_cols) == 1:
                proba_columns = list(map(
                    lambda x: '%s_%s' % (self.pred_cols[0], str(x)),
                    self.classes))
            else:
                proba_columns = self.pred_cols
            self.Y_pred_proba_df = pd.merge(
                self.Y_pred_proba_df,
                pd.DataFrame(
                    data=self.Y_pred_proba,
                    columns=proba_columns),
                left_index=True, right_index=True)
        else:
            logger.warning(
                'NOT MATCH DIMENSION OF Y_PRED_PROBA AND CLASSES')
        return

    def _calc_post_predict_df(self):
        fit_post = self.configs['post']
        if not fit_post:
            return

        if not self.kernel:
            myfunc = importlib.import_module(
                'modules.myfuncs.%s' % fit_post['myfunc'])
        for method_name in fit_post['methods']:
            logger.info('fit post: %s' % method_name)
            if not self.kernel:
                method_name = 'myfunc.%s' % method_name
            self.Y_pred_df, self.Y_pred_proba_df = eval(
                method_name)(self.Y_pred_df, self.Y_pred_proba_df)
        return

    def _round_predict_df(self):
        if isinstance(self.Y_pred_df, pd.DataFrame):
            self.Y_pred_df = self.Y_pred_df.round(5)
        if isinstance(self.Y_pred_proba_df, pd.DataFrame):
            self.Y_pred_proba_df = self.Y_pred_proba_df.round(5)
        return

    def calc_predict_data(self):
        self.Y_pred, self.Y_pred_proba = self.predict_like(
            train_mode=self.configs['fit']['train_mode'],
            estimator=self.estimator, X_train=self.X_train,
            Y_train=self.Y_train, X_target=self.X_test)
        self._inverse_translate_y_pre()
        self._calc_base_predict_df()
        self._calc_post_predict_df()
        self._round_predict_df()
        return self.Y_pred_df, self.Y_pred_proba_df

    def write_predict_data(self):
        modelname = self.configs['fit'].get('modelname', 'tmp_model')
        filename = '%s.csv' % modelname
        output_path = self.configs['data']['output_dir']
        if isinstance(self.Y_pred_df, pd.DataFrame):
            self.Y_pred_df.to_csv(
                '%s/%s' % (output_path, filename), index=False)
        if isinstance(self.Y_pred_proba_df, pd.DataFrame):
            self.Y_pred_proba_df.to_csv(
                '%s/proba_%s' % (output_path, filename), index=False)
        return filename
