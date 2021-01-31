import copy
import importlib
import os
from logging import getLogger

import dill

import numpy as np

from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from skorch import NeuralNetClassifier, NeuralNetRegressor


logger = getLogger('predict').getChild('Trainer')
if 'BaseTrainer' not in globals():
    from .BaseTrainer import BaseTrainer
if 'SingleTrainer' not in globals():
    from .SingleTrainer import SingleTrainer
if 'EnsembleTrainer' not in globals():
    from .EnsembleTrainer import EnsembleTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        feature_columns, train_ids, test_ids,
        X_train, Y_train, X_test,
        kernel=False
    ):
        self.feature_columns = feature_columns
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.kernel = kernel
        self.configs = {}

    @classmethod
    def get_cvs_from_json(self, cv_config):
        if not cv_config:
            train_cv = KFold(
                n_splits=3, shuffle=True, random_state=42)
            val_cv = copy.deepcopy(train_cv)
            val_cv.random_state = 43
            return train_cv, val_cv

        fold = cv_config['fold']
        num = cv_config['num']
        if num == 1:
            cv = 1
            return cv, cv

        if fold == 'timeseries':
            train_cv = val_cv = TimeSeriesSplit(n_splits=num)
        elif fold == 'k':
            train_cv = KFold(
                n_splits=num, shuffle=True, random_state=42)
            val_cv = copy.deepcopy(train_cv)
            val_cv.random_state = 43
        elif fold == 'stratifiedk':
            train_cv = StratifiedKFold(
                n_splits=num, shuffle=True, random_state=42)
            val_cv = copy.deepcopy(train_cv)
            val_cv.random_state = 43
        else:
            logger.error(f'NOT IMPLEMENTED CV: {fold}')
            raise Exception('NOT IMPLEMENTED')
        return train_cv, val_cv

    def _get_scorer_from_string(self, scoring):
        if scoring == 'my_scorer':
            if not self.kernel:
                myfunc = importlib.import_module(
                    'modules.myfuncs.%s'
                    % self.configs['fit'].get('myfunc'))
            method_name = 'get_my_scorer'
            if not self.kernel:
                method_name = 'myfunc.%s' % method_name
            scorer = eval(method_name)()
        else:
            scorer = get_scorer(scoring)
        return scorer

    def get_estimator_data(self):
        output = {
            'train_cv': self.train_cv,
            'val_cv': self.val_cv,
            'scorer': self.scorer,
            'classes': self.classes,
            'single_estimators': self.single_estimators,
            'estimator': self.estimator,
        }
        return output

    def calc_estimator_data(self):
        # configs
        model_configs = self.configs['fit']['single_model_configs']
        self.train_cv, self.val_cv = \
            self.get_cvs_from_json(self.configs['fit'].get('cv'))
        logger.info(f'cvs: {self.train_cv} {self.val_cv}')
        logger.info('scoring: %s' % self.configs['fit']['scoring'])
        self.scorer = self._get_scorer_from_string(
            self.configs['fit']['scoring'])
        myfunc = self.configs['fit'].get('myfunc')
        self.classes = None

        # single
        logger.info('single fit')
        single_scores = []
        self.single_estimators = []
        single_trainer_obj = SingleTrainer(
            self.X_train, self.Y_train, self.X_test, self.kernel)
        single_trainer_obj.configs = self.configs
        for i, config in enumerate(model_configs):
            _score, _estimator = single_trainer_obj.calc_single_estimator(
                config, self.scorer,
                self.train_cv, self.val_cv, nn_func=myfunc)
            single_scores.append(_score)
            modelname = f'{i}_{config["model"]}'
            self.single_estimators.append(
                (modelname, _estimator))

        # ensemble
        if len(self.single_estimators) == 1:
            logger.info('no ensemble')
            self.estimator = self.single_estimators[0][1]
        else:
            ensemble_trainer_obj = EnsembleTrainer(
                self.X_train, self.Y_train, self.X_test)
            ensemble_trainer_obj.configs = self.configs
            self.estimator = ensemble_trainer_obj.calc_ensemble_estimator(
                self.single_estimators,
                weights=EnsembleTrainer.get_weights(single_scores),
                scorer=self.scorer)

        # classes
        if self.configs['fit']['train_mode'] == 'clf':
            for _, single_estimator in self.single_estimators:
                if self.classes is not None:
                    continue
                if hasattr(single_estimator, 'classes_'):
                    self.classes = single_estimator.classes_
                else:
                    self.classes = np.sort(np.unique(self.Y_train))
        return self.estimator

    def write_estimator_data(self):
        modelname = self.configs['fit'].get('modelname', 'tmp_model')
        if len(self.single_estimators) == 1:
            targets = self.single_estimators
        else:
            targets = self.single_estimators + [
                (modelname, self.estimator)
            ]
        for modelname, estimator in targets:
            output_path = self.configs['data']['output_dir']
            _savename = f'{output_path}/{modelname}'

            if estimator.__class__ in [
                NeuralNetClassifier, NeuralNetRegressor
            ]:
                logger.warning('NOT IMPLEMENTED TORCH MODEL SAVE')
            elif hasattr(estimator, 'save'):
                estimator.save(f'{_savename}.pickle')
            # keras
            elif hasattr(estimator, 'model'):
                if hasattr(estimator.model, 'save'):
                    os.makedirs(f'{_savename}', exist_ok=True)
                    estimator.model.save(f'{_savename}')
            else:
                with open(f'{_savename}.pickle', 'wb') as f:
                    dill.dump(estimator, f)
        return modelname
