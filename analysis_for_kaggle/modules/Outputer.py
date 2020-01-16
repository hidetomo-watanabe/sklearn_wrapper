import math
import numpy as np
import pandas as pd
import importlib
from keras.engine.sequential import Sequential
from sklearn.ensemble import VotingClassifier
from heamy.dataset import Dataset
from heamy.estimator import Classifier, Regressor
from heamy.pipeline import PipeApply
from sklearn.metrics import confusion_matrix
from logging import getLogger


logger = getLogger('predict').getChild('Outputer')
try:
    from .ConfigReader import ConfigReader
except ImportError:
    logger.warning('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')


class Outputer(ConfigReader):
    def __init__(
        self,
        feature_columns, test_ids,
        X_train, Y_train, X_test,
        cv, scorer, classes, single_estimators, estimator,
        y_scaler=None, kernel=False
    ):
        self.feature_columns = feature_columns
        self.test_ids = test_ids
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.y_scaler = y_scaler
        self.classes = classes
        self.estimator = estimator
        self.kernel = kernel
        self.configs = {}

    def get_predict_data(self):
        output = {
            'Y_pred': self.Y_pred,
            'Y_pred_proba': self.Y_pred_proba,
            'Y_train_pred': self.Y_train_pred,
            'Y_train_pred_proba': self.Y_train_pred_proba,
            'Y_pred_df': self.Y_pred_df,
            'Y_pred_proba_df': self.Y_pred_proba_df,
        }
        return output

    def predict_y(self):
        self.Y_pred = None
        self.Y_pred_proba = None
        self.Y_train_pred = None
        self.Y_train_pred_proba = None
        # clf
        if self.configs['fit']['train_mode'] == 'clf':
            # keras clf
            if self.estimator.__class__ in [Sequential]:
                self.Y_pred = self.estimator.predict_classes(self.X_test)
                self.Y_train_pred = self.estimator.predict_classes(
                    self.X_train)
                self.Y_pred_proba = self.estimator.predict(self.X_test)
                self.Y_train_pred_proba = self.estimator.predict(
                    self.X_train)
            # stacker clf
            elif self.estimator.__class__ in [Classifier]:
                # no proba
                self.estimator.probability = False
                self.Y_pred = self.estimator.predict()
                # proba
                self.estimator.probability = True
                # from heamy sorce code, to make Y_pred_proba multi dimension
                self.estimator.problem = ''
                self.Y_pred_proba = self.estimator.predict()
                # train no proba
                self.estimator.probability = False
                dataset = Dataset(
                    self.X_train.toarray(), self.Y_train,
                    self.X_train.toarray())
                self.estimator.dataset = dataset
                self.Y_train_pred = self.estimator.predict()
                # train proba
                self.estimator.probability = True
                # from heamy sorce code, to make Y_pred_proba multi dimension
                self.estimator.problem = ''
                self.Y_train_pred_proba = self.estimator.predict()
            # voter clf
            elif self.estimator.__class__ in [VotingClassifier]:
                self.Y_pred = self.estimator.predict(self.X_test)
                self.Y_train_pred = self.estimator.predict(self.X_train)
                if self.estimator.voting == 'soft':
                    self.Y_pred_proba = self.estimator.predict_proba(
                        self.X_test)
                    self.Y_train_pred_proba = self.estimator.predict_proba(
                        self.X_train)
            # single clf
            else:
                self.Y_pred = self.estimator.predict(self.X_test)
                self.Y_train_pred = self.estimator.predict(self.X_train)
                if hasattr(self.estimator, 'predict_proba'):
                    self.Y_pred_proba = self.estimator.predict_proba(
                        self.X_test)
                    self.Y_train_pred_proba = self.estimator.predict_proba(
                        self.X_train)
            logger.info(
                f'confusion_matrix:'
                f' \n{confusion_matrix(self.Y_train, self.Y_train_pred)}')
        # reg
        elif self.configs['fit']['train_mode'] == 'reg':
            # weighted_average reg
            if self.estimator.__class__ in [PipeApply]:
                self.Y_pred = self.estimator.execute()
            # ensemble reg
            elif self.estimator.__class__ in [Regressor]:
                self.Y_pred = self.estimator.predict()
                # train
                dataset = Dataset(self.X_train, self.Y_train, self.X_train)
                self.estimator.dataset = dataset
                self.Y_train_pred = self.estimator.predict()
            # single reg
            else:
                self.Y_pred = self.estimator.predict(self.X_test)
                self.Y_train_pred = self.estimator.predict(self.X_train)

            # inverse normalize
            # scaler
            self.Y_pred = self.Y_pred.reshape(-1, 1)
            self.Y_pred = self.y_scaler.inverse_transform(self.Y_pred)
            if isinstance(self.Y_train_pred, np.ndarray):
                self.Y_train_pred = self.Y_train_pred.reshape(-1, 1)
                self.Y_train_pred = \
                    self.y_scaler.inverse_transform(self.Y_train_pred)
            else:
                logger.warning('NO Y_train_pred')
            # pre
            y_pre = self.configs['pre'].get('y_pre')
            if y_pre:
                logger.info('inverse translate y_train with %s' % y_pre)
                if y_pre == 'log':
                    self.Y_pred = np.array(list(map(math.exp, self.Y_pred)))
                    self.Y_train_pred = np.array(list(map(
                        math.exp, self.Y_train_pred)))
                else:
                    logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
                    raise Exception('NOT IMPLEMENTED')
        else:
            logger.error('TRAIN MODE SHOULD BE clf OR reg')
            raise Exception('NOT IMPLEMENTED')
        return \
            self.Y_pred, self.Y_pred_proba, \
            self.Y_train_pred, self.Y_train_pred_proba

    def calc_predict_df(self):
        self.Y_pred_df = None
        self.Y_pred_proba_df = None
        # np => pd
        if isinstance(self.Y_pred, np.ndarray):
            self.Y_pred_df = pd.merge(
                pd.DataFrame(data=self.test_ids, columns=[self.id_col]),
                pd.DataFrame(data=self.Y_pred, columns=self.pred_cols),
                left_index=True, right_index=True)
        if isinstance(self.Y_pred_proba, np.ndarray):
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

        # post
        fit_post = self.configs['post']
        if fit_post:
            if not self.kernel:
                myfunc = importlib.import_module(
                    'modules.myfuncs.%s' % fit_post['myfunc'])
            for method_name in fit_post['methods']:
                logger.info('fit post: %s' % method_name)
                if not self.kernel:
                    method_name = 'myfunc.%s' % method_name
                self.Y_pred_df, self.Y_pred_proba_df = eval(
                    method_name)(self.Y_pred_df, self.Y_pred_proba_df)

        return self.Y_pred_df, self.Y_pred_proba_df

    def write_predict_data(self):
        modelname = \
            self.configs['fit']['ensemble_model_config'].get('modelname')
        if not modelname:
            modelname = 'tmp_model'
        filename = '%s.csv' % modelname
        output_path = self.configs['data']['output_dir']
        if isinstance(self.Y_pred_df, pd.DataFrame):
            self.Y_pred_df.round(5).to_csv(
                '%s/%s' % (output_path, filename), index=False)
        if isinstance(self.Y_pred_proba_df, pd.DataFrame):
            self.Y_pred_proba_df.round(5).to_csv(
                '%s/proba_%s' % (output_path, filename), index=False)
        return filename
