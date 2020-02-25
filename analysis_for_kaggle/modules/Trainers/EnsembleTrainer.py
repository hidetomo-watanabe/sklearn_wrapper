import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from heamy.dataset import Dataset
from heamy.estimator import Classifier, Regressor
from heamy.pipeline import ModelsPipeline
from sklearn.metrics import get_scorer
from IPython.display import display
from logging import getLogger


logger = getLogger('predict').getChild('EnsembleTrainer')
if 'BaseTrainer' not in globals():
    from .BaseTrainer import BaseTrainer


class EnsembleTrainer(BaseTrainer):
    def __init__(self, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.configs = {}

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

    def _get_pipeline(self, single_estimators):
        # for warning
        Y_train = self.ravel_like(self.Y_train)
        dataset = Dataset(
            self.toarray_like(self.X_train), Y_train,
            self.toarray_like(self.X_test))
        models = []
        for modelname, single_estimator in single_estimators:
            # clf
            if self.configs['fit']['train_mode'] == 'clf':
                models.append(
                    Classifier(
                        dataset=dataset, estimator=single_estimator.__class__,
                        parameters=single_estimator.get_params(),
                        name=modelname))
            # reg
            elif self.configs['fit']['train_mode'] == 'reg':
                models.append(
                    Regressor(
                        dataset=dataset, estimator=single_estimator.__class__,
                        parameters=single_estimator.get_params(),
                        name=modelname))
        pipeline = ModelsPipeline(*models)
        return pipeline

    def _get_stacker(self, pipeline, ensemble_config):
        if ensemble_config['mode'] == 'stacking':
            stack_dataset = pipeline.stack(
                k=ensemble_config['k'], seed=42)
        elif ensemble_config['mode'] == 'blending':
            stack_dataset = pipeline.blend(
                proportion=ensemble_config['proportion'], seed=42)
        if self.configs['fit']['train_mode'] == 'clf':
            stacker = Classifier(
                dataset=stack_dataset,
                estimator=self.get_base_model(
                    ensemble_config['model']).__class__)
        elif self.configs['fit']['train_mode'] == 'reg':
            stacker = Regressor(
                dataset=stack_dataset,
                estimator=self.get_base_model(
                    ensemble_config['model']).__class__)
        stacker.use_cache = False
        # default predict
        stacker.probability = False
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
            voter = self._get_voter(
                ensemble_config['mode'], single_estimators, weights)
            Y_train = self.ravel_like(Y_train)
            voter.fit(X_train, Y_train)
            estimator = voter
        elif ensemble_config['mode'] in ['stacking', 'blending']:
            pipeline = self._get_pipeline(single_estimators)
            stacker = self._get_stacker(pipeline, ensemble_config)
            stacker.validate(
                k=ensemble_config['k'], scorer=scorer._score_func)
            estimator = stacker
        else:
            logger.error(
                'NOT IMPLEMENTED ENSEMBLE MODE: %s' % ensemble_config['mode'])
            raise Exception('NOT IMPLEMENTED')
        return estimator
