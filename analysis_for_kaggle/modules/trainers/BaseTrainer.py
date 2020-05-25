import copy
import types
from logging import getLogger

from bert_sklearn import BertClassifier, BertRegressor

from catboost import CatBoostClassifier, CatBoostRegressor

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from lightgbm import LGBMClassifier, LGBMRegressor

import numpy as np

import optuna

from rgf.sklearn import RGFClassifier, RGFRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skorch import NeuralNetClassifier, NeuralNetRegressor

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Sequential

import torch

from xgboost import XGBClassifier, XGBRegressor


logger = getLogger('predict').getChild('BaseTrainer')
if 'ConfigReader' not in globals():
    from ..ConfigReader import ConfigReader
if 'Flattener' not in globals():
    from ..commons.Flattener import Flattener
if 'Reshaper' not in globals():
    from ..commons.Reshaper import Reshaper
if 'LikeWrapper' not in globals():
    from ..commons.LikeWrapper import LikeWrapper
if 'Augmentor' not in globals():
    from .Augmentor import Augmentor


class MyKerasClassifier(KerasClassifier):
    def fit(
        self, x, y,
        with_generator=False, generator=None, batch_size=None, validation_data=None,
        **kwargs
    ):
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType):
            if not isinstance(self.build_fn, types.MethodType):
                self.model = self.build_fn(
                    **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        if losses.is_categorical_crossentropy(self.model.loss):
            if len(y.shape) != 2:
                y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)

        if with_generator:
            logger.info(f'with generator')
            history = self.model.fit_generator(
                generator.flow(x, y, batch_size=batch_size),
                validation_data=generator.flow(
                    validation_data[0], validation_data[1],
                    batch_size=batch_size),
                **kwargs)
        else:
            history = self.model.fit(x, y, **fit_args)
        return history


class MyKerasRegressor(KerasRegressor):
    def fit(
        self, x, y,
        with_generator=False, generator=None, batch_size=None, validation_data=None,
        **kwargs
    ):
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType):
            if not isinstance(self.build_fn, types.MethodType):
                self.model = self.build_fn(
                    **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        if losses.is_categorical_crossentropy(self.model.loss):
            if len(y.shape) != 2:
                y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        if with_generator:
            logger.info(f'with generator')
            history = self.model.fit_generator(
                generator.flow(x, y, batch_size=batch_size),
                validation_data=generator.flow(
                    validation_data[0], validation_data[1],
                    batch_size=batch_size),
                **kwargs)
        else:
            history = self.model.fit(x, y, **fit_args)
        return history


class BaseTrainer(ConfigReader, LikeWrapper):
    def __init__(self):
        pass

    @classmethod
    def get_base_estimator(self, model, create_nn_model=None):
        # keras config
        tf.random.set_seed(42)

        # torch config
        # for reproducibility
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # gpu or cpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model == 'log_reg':
            return LogisticRegression(solver='lbfgs')
        elif model == 'log_reg_cv':
            return LogisticRegressionCV()
        elif model == 'linear_reg':
            return LinearRegression()
        elif model == 'lasso':
            return Lasso()
        elif model == 'ridge':
            return Ridge()
        elif model == 'svc':
            return SVC()
        elif model == 'svr':
            return SVR()
        elif model == 'l_svc':
            return LinearSVC()
        elif model == 'l_svr':
            return LinearSVR()
        elif model == 'rf_clf':
            return RandomForestClassifier()
        elif model == 'rf_reg':
            return RandomForestRegressor()
        elif model == 'gbdt_clf':
            return GradientBoostingClassifier()
        elif model == 'gbdt_reg':
            return GradientBoostingRegressor()
        elif model == 'knn_clf':
            return KNeighborsClassifier()
        elif model == 'knn_reg':
            return KNeighborsRegressor()
        elif model == 'g_mix':
            return GaussianMixture()
        elif model == 'g_nb':
            return GaussianNB()
        elif model == 'preceptron':
            return Perceptron()
        elif model == 'sgd_clf':
            return SGDClassifier()
        elif model == 'sgd_reg':
            return SGDRegressor()
        elif model == 'dt_clf':
            return DecisionTreeClassifier()
        elif model == 'dt_reg':
            return DecisionTreeRegressor()
        elif model == 'xgb_clf':
            return XGBClassifier()
        elif model == 'xgb_reg':
            return XGBRegressor()
        elif model == 'lgb_clf':
            return LGBMClassifier()
        elif model == 'lgb_reg':
            return LGBMRegressor()
        elif model == 'catb_clf':
            return CatBoostClassifier()
        elif model == 'catb_reg':
            return CatBoostRegressor()
        elif model == 'rgf_clf':
            return RGFClassifier()
        elif model == 'rgf_reg':
            return RGFRegressor()
        elif model == 'keras_clf':
            return MyKerasClassifier(build_fn=create_nn_model)
        elif model == 'keras_reg':
            return MyKerasRegressor(build_fn=create_nn_model)
        elif model == 'torch_clf':
            return NeuralNetClassifier(
                module=create_nn_model(), device=device, train_split=None)
        elif model == 'torch_reg':
            return NeuralNetRegressor(
                module=create_nn_model(), device=device, train_split=None)
        elif model == 'bert_clf':
            return BertClassifier()
        elif model == 'bert_reg':
            return BertRegressor()
        else:
            logger.error('NOT IMPLEMENTED BASE MODEL: %s' % model)
            raise Exception('NOT IMPLEMENTED')

    @classmethod
    def to_second_estimator(
        self, estimator, multiclass=None, undersampling=None
    ):
        if multiclass:
            estimator = multiclass(estimator=estimator, n_jobs=-1)
        if undersampling:
            estimator = undersampling(
                base_estimator=estimator, random_state=42, n_jobs=-1)
        # no oversampling
        return estimator

    @classmethod
    def _trans_xy_for_fit(self, estimator, X_train, Y_train):
        for step in estimator.steps:
            if step[1].__class__ in [MyKerasClassifier]:
                if self.ravel_like(Y_train).ndim == 1:
                    Y_train = to_categorical(Y_train)
            elif step[1].__class__ in [BertClassifier, BertRegressor]:
                X_train = self.ravel_like(X_train)
        Y_train = self.ravel_like(Y_train)
        return X_train, Y_train

    @classmethod
    def _trans_step_for_fit(self, estimator, X_train, Y_train):
        is_categorical = False
        indexes = []
        for i, step in enumerate(estimator.steps):
            if step[1].__class__ in [
                RandomUnderSampler, RandomOverSampler, SMOTE
            ]:
                indexes.append(i)
            elif step[1].__class__ in [MyKerasClassifier]:
                if self.ravel_like(Y_train).ndim == 1:
                    is_categorical = True

        base = 0
        for i in indexes:
            estimator.steps.insert(
                i + base,
                ('flattener', Flattener()))
            estimator.steps.insert(
                i + base + 2,
                ('reshaper', Reshaper(X_train.shape[1:], is_categorical)))
            base += 2
        return estimator

    @classmethod
    def _add_val_to_fit_params(self, fit_params, estimator, X_test, Y_test):
        aug_index = None
        aug_obj = None
        for i, step in enumerate(estimator.steps):
            if step[1].__class__ in [Augmentor]:
                aug_index = i
                aug_obj = step[1]

        for step in estimator.steps:
            if step[1].__class__ in [MyKerasClassifier, MyKerasRegressor]:
                if aug_obj:
                    fit_params[f'{step[0]}__with_generator'] = True
                    fit_params[f'{step[0]}__generator'] = aug_obj.datagen
                    fit_params[f'{step[0]}__batch_size'] = aug_obj.batch_size
                    fit_params[f'{step[0]}__steps_per_epoch'] = aug_obj.steps
                    fit_params[f'{step[0]}__validation_data'] = \
                        (X_test, Y_test)
                    fit_params[f'{step[0]}__validation_steps'] = aug_obj.steps
                    step = step[: aug_index] + step[aug_index + 1:]
                else:
                    fit_params[f'{step[0]}__validation_data'] = \
                        (X_test, Y_test)
            elif step[1].__class__ in [LGBMClassifier, LGBMRegressor]:
                if aug_obj:
                    X_test, Y_test = aug_obj.fit_resample(X_test, Y_test)
                fit_params[f'{step[0]}__eval_set'] = [(X_test, Y_test)]
        return fit_params

    @classmethod
    def calc_cv_scores_estimators(
        self, estimator, X_train, Y_train, scorer, cv, fit_params
    ):
        scores = []
        estimators = []
        if cv == 1:
            indexes = [[range(X_train.shape[0]), range(X_train.shape[0])]]
        else:
            indexes = cv.split(X_train, Y_train)

        X_train_for_fit, Y_train_for_fit = \
            self._trans_xy_for_fit(estimator, X_train, Y_train)
        estimator = self._trans_step_for_fit(estimator, X_train, Y_train)
        for train_index, test_index in indexes:
            fit_params = self._add_val_to_fit_params(
                fit_params, estimator,
                X_train_for_fit[test_index], Y_train_for_fit[test_index])
            tmp_estimator = estimator
            tmp_estimator.fit(
                X_train_for_fit[train_index], Y_train_for_fit[train_index],
                **fit_params)
            estimators.append(tmp_estimator)
            scores.append(scorer(
                tmp_estimator,
                X_train_for_fit[test_index], Y_train[test_index]))
        return scores, estimators

    @classmethod
    def calc_best_params(
        self,
        base_estimator, X_train, Y_train, params, scorer, cv, fit_params,
        n_trials=None, multiclass=None, undersampling=None
    ):
        def _get_args(trial, params):
            args = {}
            for key, val in params.items():
                if isinstance(val, list):
                    args[key] = trial.suggest_categorical(key, val)
                elif isinstance(val, dict):
                    if val['type'] == 'int':
                        args[key] = trial.suggest_int(
                            key, val['from'], val['to'])
                    elif val['type'] == 'float':
                        args[key] = trial.suggest_uniform(
                            key, val['from'], val['to'])
                    elif val['type'] == 'log':
                        args[key] = trial.suggest_loguniform(
                            key, val['from'], val['to'])
                    else:
                        logger.error(
                            f'ILLEGAL PARAM TYPE ON {key}: {val["type"]}')
                        raise Exception('ILLEGAL VALUE')
                else:
                    logger.error(
                        f'ILLEGAL DATA TYPE ON {key}: {type(val)}')
                    raise Exception('ILLEGAL VALUE')
            return args

        def _objective(trial):
            args = _get_args(trial, params)
            estimator = base_estimator
            estimator.set_params(**args)
            estimator = self.to_second_estimator(
                estimator, multiclass, undersampling)
            try:
                scores, _ = self.calc_cv_scores_estimators(
                    estimator, X_train, Y_train, scorer, cv, fit_params)
            except Exception as e:
                logger.warning(e)
                logger.warning('SET SCORE 0')
                scores = [0]
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            logger.debug('  params: %s' % args)
            logger.debug('  scores: %s' % scores)
            logger.debug('  score mean: %s' % score_mean)
            logger.debug('  score std: %s' % score_std)
            return -1 * score_mean

        all_comb_num = 0
        for val in params.values():
            if isinstance(val, list):
                if all_comb_num == 0:
                    all_comb_num = 1
                all_comb_num *= len(val)
            elif isinstance(val, dict):
                all_comb_num = None
                break
        # no search
        if all_comb_num in [0, 1]:
            logger.info(f'no search because params pattern is {all_comb_num}')
            if all_comb_num == 0:
                return {}
            elif all_comb_num == 1:
                single_param = {}
                for key, val in params.items():
                    single_param[key] = val[0]
                return single_param

        if all_comb_num:
            if n_trials is None:
                n_trials = all_comb_num
            elif n_trials > all_comb_num:
                logger.warning(
                    f'N_TRIALS IS OVER MAX PATTERN THEN SET WITH MAX')
                n_trials = all_comb_num
            elif n_trials < 0:
                logger.error(f'N_TRIALS SHOULD BE LARGER THAN 0: {n_trials}')
                raise Exception('ILLEGAL VALUE')
        else:
            if n_trials is None:
                logger.error(f'IF PARAMS HAVE DICT, N_TRIALS SHOULD BE SET')
                raise Exception('ILLEGAL VALUE')
        logger.info(f'n_trials: {n_trials}')

        study = optuna.create_study()
        study.optimize(_objective, n_trials=n_trials)
        best_params = study.best_params
        best_score_mean = -1 * study.best_trial.value
        logger.info('best score mean: %s' % best_score_mean)
        return best_params
