from logging import getLogger

from IPython.display import display

import numpy as np

import pandas as pd

from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier, VotingRegressor
)
from sklearn.metrics import get_scorer


logger = getLogger('predict').getChild('EnsembleTrainer')
if 'BaseTrainer' not in globals():
    from .BaseTrainer import BaseTrainer


class EnsembleTrainer(BaseTrainer):
    def __init__(self, X_train, Y_train, X_test, configs):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.configs = configs

    def _get_voter(self, mode, estimators, weights=None):
        if self.configs['fit']['train_mode'] == 'clf':
            if mode == 'average':
                voting = 'soft'
            elif mode == 'vote':
                voting = 'hard'
            voter = VotingClassifier(
                estimators=estimators, voting=voting,
                weights=weights, n_jobs=-1)
        elif self.configs['fit']['train_mode'] == 'reg':
            if mode == 'average':
                voter = VotingRegressor(
                    estimators=estimators, weights=weights, n_jobs=-1)
        return voter

    def _get_stacker(self, mode, estimators, ensemble_config):
        if self.configs['fit']['train_mode'] == 'clf':
            stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=self.get_base_estimator(
                    ensemble_config['model']),
                n_jobs=-1)
        elif self.configs['fit']['train_mode'] == 'reg':
            stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=self.get_base_estimator(
                    ensemble_config['model']),
                n_jobs=-1)
        return stacker

    @classmethod
    def get_weights(self, scores):
        scores = np.array(scores)
        return scores / np.sum(scores)

    def calc_ensemble_estimator(
        self, single_estimators, ensemble_config=None, weights=None,
        scorer=get_scorer('accuracy'), X_train=None, Y_train=None
    ):
        if ensemble_config is None:
            ensemble_config = self.configs['fit']['ensemble_model_config']
        if X_train is None:
            X_train = self.X_train
        if Y_train is None:
            Y_train = self.Y_train

        logger.info('ensemble fit: %s' % ensemble_config['mode'])
        if ensemble_config['mode'] in ['average', 'vote']:
            if ensemble_config['mode'] == 'vote' \
                    and self.configs['fit']['train_mode'] == 'reg':
                logger.error(
                    'NOT IMPLEMENTED REGRESSION AND VOTE')
                raise Exception('NOT IMPLEMENTED')

            logger.info('weights:')
            display(pd.DataFrame(
                    weights.reshape(-1, weights.shape[0]),
                    columns=[_e[0] for _e in single_estimators]))
            estimator = self._get_voter(
                ensemble_config['mode'], single_estimators, weights)
        elif ensemble_config['mode'] in ['stacking']:
            estimator = self._get_stacker(
                ensemble_config['mode'], single_estimators, ensemble_config)
        else:
            logger.error(
                'NOT IMPLEMENTED ENSEMBLE MODE: %s' % ensemble_config['mode'])
            raise Exception('NOT IMPLEMENTED')

        Y_train = self.ravel_like(Y_train)
        estimator.fit(X_train, Y_train)
        return estimator
