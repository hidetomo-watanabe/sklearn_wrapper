import scipy.sparse as sp
import numpy as np
import importlib
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from sklearn.metrics import get_scorer
from logging import getLogger


logger = getLogger('predict').getChild('SingleTrainer')
if 'BaseTrainer' not in globals():
    from .BaseTrainer import BaseTrainer
if 'EnsembleTrainer' not in globals():
    from .EnsembleTrainer import EnsembleTrainer
if 'Outputer' not in globals():
    from ..Outputer import Outputer


class SingleTrainer(BaseTrainer):
    def __init__(self, X_train, Y_train, X_test, kernel=False):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.kernel = kernel
        self.configs = {}
        # keras, torchのスコープ対策として、インスタンス作成時に読み込み
        # keras, torch使う時しか使わないので、evalで定義してエラー回避
        if self.kernel:
            self.create_nn_model = eval('create_nn_model')

    def _get_model_params(self, model_config, nn_func, X_train, Y_train):
        model = model_config['model']
        logger.info('model: %s' % model)
        self.model = model
        modelname = model_config.get('modelname')
        if modelname:
            logger.info('modelname: %s' % modelname)
        create_nn_model = None
        if model in ['keras_clf', 'keras_reg', 'torch_clf', 'torch_reg']:
            if self.kernel:
                create_nn_model = self.create_nn_model
            else:
                myfunc = importlib.import_module(
                    'modules.myfuncs.%s' % nn_func)
                create_nn_model = myfunc.create_nn_model
        self.base_estimator = self.get_base_estimator(
            model, create_nn_model=create_nn_model)
        multiclass = model_config.get('multiclass')
        if multiclass:
            logger.info('multiclass: %s' % multiclass)
            if multiclass == 'onevsone':
                multiclass = OneVsOneClassifier
            elif multiclass == 'onevsrest':
                multiclass = OneVsRestClassifier
            else:
                logger.error(
                    f'NOT IMPLEMENTED MULTICLASS: {multiclass}')
                raise Exception('NOT IMPLEMENTED')
        self.multiclass = multiclass
        undersampling = model_config.get('undersampling')
        if undersampling:
            logger.info(f'undersampling: {undersampling}')
            if undersampling == 'bagging':
                undersampling = BalancedBaggingClassifier
            elif undersampling == 'adaboost':
                undersampling = RUSBoostClassifier
        self.undersampling = undersampling
        self.cv_select = model_config.get('cv_select', 'nearest_mean')
        self.n_trials = model_config.get('n_trials')
        fit_params = model_config.get('fit_params', {})
        if model in ['lgb_clf', 'lgb_reg']:
            fit_params['eval_set'] = [(X_train, Y_train)]
        self.fit_params = fit_params
        self.params = model_config.get('params', {})
        return

    def _fit(self, scorer, cv, X_train, Y_train):
        best_params = self.calc_best_params(
            self.base_estimator, X_train, Y_train, self.params,
            scorer, cv, self.fit_params, self.n_trials,
            self.multiclass, self.undersampling)
        logger.info('best params: %s' % best_params)
        estimator = self.base_estimator
        estimator.set_params(**best_params)
        estimator = self.to_second_estimator(
            estimator, self.multiclass, self.undersampling)
        logger.info(f'get estimator with cv_select: {self.cv_select}')
        if self.cv_select == 'train_all':
            scores, estimators = self.calc_cv_scores_estimators(
                estimator, X_train, Y_train, scorer,
                cv=1, fit_params=self.fit_params)
        elif self.cv_select in ['nearest_mean', 'all_folds']:
            scores, estimators = self.calc_cv_scores_estimators(
                estimator, X_train, Y_train, scorer,
                cv=cv, fit_params=self.fit_params)
            logger.info(f'cv model scores mean: {np.mean(scores)}')
            logger.info(f'cv model scores std: {np.std(scores)}')
            if self.cv_select == 'nearest_mean':
                nearest_index \
                    = np.abs(np.array(scores) - np.mean(scores)).argmin()
                scores = scores[nearest_index: nearest_index + 1]
                estimators = estimators[nearest_index: nearest_index + 1]
            elif self.cv_select == 'all_folds':
                _single_estimators = []
                for i, _estimator in enumerate(estimators):
                    _single_estimators.append(
                        (f'{i}_fold', _estimator))
                weights = EnsembleTrainer.get_weights(scores)

                ensemble_trainer_obj = EnsembleTrainer(
                    X_train, Y_train, self.X_test)
                ensemble_trainer_obj.configs = self.configs
                estimator = ensemble_trainer_obj.calc_ensemble_estimator(
                    _single_estimators, ensemble_config={'mode': 'average'},
                    weights=weights, scorer=scorer)
                scores = [np.average(scores, weights=weights)]
                estimators = [estimator]
        else:
            logger.error(f'NOT IMPLEMENTED CV SELECT: {cv_select}')
            raise Exception('NOT IMPLEMENTED')
        return scores[0], estimators[0]

    def _calc_pseudo_label_data(
        self, X_train, Y_train, estimator, classes, threshold
    ):
        _, Y_pred_proba = Outputer.predict_like(
            train_mode=self.configs['fit']['train_mode'],
            estimator=estimator, X_train=X_train, Y_train=Y_train,
            X_target=self.X_test)

        data_index, label_index = np.where(Y_pred_proba > threshold)
        pseudo_X_train = self.X_test[data_index]
        pseudo_Y_train = classes[label_index]
        return pseudo_X_train, pseudo_Y_train

    def _fit_with_pseudo_labeling(
        self, scorer, cv, estimator, X_train, Y_train, classes, threshold
    ):
        logger.info('fit with pseudo labeling')
        pseudo_X_train, pseudo_Y_train = self._calc_pseudo_label_data(
            X_train, Y_train, estimator, classes, threshold)
        new_X_train = sp.vstack((X_train, pseudo_X_train), format='csr')
        new_Y_train = np.concatenate([Y_train, pseudo_Y_train])
        logger.info(
            'with threshold %s, train data added %s => %s'
            % (threshold, len(Y_train), len(new_Y_train)))
        return self._fit(scorer, cv, new_X_train, new_Y_train)

    def _sample_with_error(self, X_train, Y_train, estimator):
        Y_pred, _ = Outputer.predict_like(
            train_mode=self.configs['fit']['train_mode'],
            estimator=estimator, X_train=X_train, Y_train=Y_train,
            X_target=X_train)

        data_index = np.where(Y_pred != Y_train)
        error_X_train = X_train[data_index]
        error_Y_train = Y_train[data_index]
        return error_X_train, error_Y_train

    def _fit_with_error_sampling(
        self, scorer, cv, estimator, X_train, Y_train, score
    ):
        logger.info('fit with error_sampling')
        new_X_train, new_Y_train = self._sample_with_error(
            X_train, Y_train, estimator)
        logger.info(
            'with error_sampling, error train data is %s'
            % len(new_Y_train))
        _score, _estimator = self._fit(scorer, cv, new_X_train, new_Y_train)

        _single_estimators = [
            ('base', estimator),
            ('error', _estimator),
        ]
        weights = EnsembleTrainer.get_weights(
            np.array([len(Y_train), len(new_Y_train)]))

        score = np.average(np.array([score, _score]), weights=weights)
        ensemble_trainer_obj = EnsembleTrainer(
            X_train, Y_train, self.X_test)
        ensemble_trainer_obj.configs = self.configs
        estimator = ensemble_trainer_obj.calc_ensemble_estimator(
            _single_estimators, ensemble_config={'mode': 'average'},
            weights=weights, scorer=scorer)
        return score, estimator

    def calc_single_estimator(
        self,
        model_config, scorer=get_scorer('accuracy'),
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        nn_func=None, X_train=None, Y_train=None
    ):
        if X_train is None:
            X_train = self.X_train
        if Y_train is None:
            Y_train = self.Y_train
        self._get_model_params(model_config, nn_func, X_train, Y_train)
        # for warning
        if self.model not in ['keras_reg', 'torch_reg']:
            Y_train = self.ravel_like(Y_train)

        # fit
        logger.info('fit')
        score, estimator = self._fit(scorer, cv, X_train, Y_train)
        logger.info(f'score: {score}')
        logger.info(f'estimator: {estimator}')

        # pseudo labeling
        pseudo_config = model_config.get('pseudo_labeling')
        if pseudo_config:
            if self.configs['fit']['train_mode'] == 'reg':
                logger.error('NOT IMPLEMENTED PSEUDO LABELING WITH REGRESSION')
                raise Exception('NOT IMPLEMENTED')

            threshold = pseudo_config.get('threshold')
            if not threshold and int(threshold) != 0:
                threshold = 0.8
            if hasattr(estimator, 'classes_'):
                classes = estimator.classes_
            else:
                classes = sorted(np.unique(Y_train))

            score, estimator = self._fit_with_pseudo_labeling(
                scorer, cv, estimator, X_train, Y_train, classes, threshold)
            logger.info(f'score: {score}')
            logger.info(f'estimator: {estimator}')

        # error sampling
        if model_config.get('error_sampling'):
            if self.configs['fit']['train_mode'] == 'reg':
                logger.error('NOT IMPLEMENTED ERROR SAMPLING WITH REGRESSION')
                raise Exception('NOT IMPLEMENTED')

            score, estimator = self._fit_with_error_sampling(
                scorer, cv, estimator, X_train, Y_train, score)
            logger.info(f'score: {score}')
            logger.info(f'estimator: {estimator}')

        return score, estimator
