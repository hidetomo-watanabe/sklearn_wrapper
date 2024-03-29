import copy
from logging import getLogger

from catboost import CatBoostClassifier, CatBoostRegressor

import eli5
from eli5.sklearn import PermutationImportance

from keras.utils.np_utils import to_categorical

from lightgbm import LGBMClassifier, LGBMRegressor

import numpy as np

import optuna

import pandas as pd

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

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

import torch

from xgboost import XGBClassifier, XGBRegressor


logger = getLogger('predict').getChild('BaseTrainer')
if 'ConfigReader' not in globals():
    from ..ConfigReader import ConfigReader
if 'LikeWrapper' not in globals():
    from ..commons.LikeWrapper import LikeWrapper
if 'MyKerasClassifier' not in globals():
    from ..commons.MyKeras import MyKerasClassifier
if 'MyKerasRegressor' not in globals():
    from ..commons.MyKeras import MyKerasRegressor
if 'Augmentor' not in globals():
    from .Augmentor import Augmentor


class BaseTrainer(ConfigReader, LikeWrapper):
    def __init__(self):
        pass

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
        elif model == 'tabnet_clf':
            return TabNetClassifier()
        elif model == 'tabnet_reg':
            return TabNetRegressor()
        else:
            logger.error('NOT IMPLEMENTED BASE MODEL: %s' % model)
            raise Exception('NOT IMPLEMENTED')

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

    def _trans_xy_for_fit(self, pipeline, X_train, Y_train):
        for step in pipeline.steps:
            # keras clf is categorical
            if step[1].__class__ in [MyKerasClassifier]:
                if self.ravel_like(Y_train).ndim == 1:
                    Y_train = to_categorical(Y_train)
                    break
            # tabnet reg is no ravel
            elif step[1].__class__ in [TabNetRegressor]:
                break
        else:
            Y_train = self.ravel_like(Y_train)
        return X_train, Y_train

    def _add_augmentor_to_fit_params(self, pipeline, fit_params):
        steps = pipeline.steps

        aug_index = None
        aug_obj = None
        for i, step in enumerate(steps):
            if step[1].__class__ in [Augmentor]:
                aug_index = i
                aug_obj = step[1]

        if not aug_index:
            return pipeline, fit_params

        # excepting keras, as it is pipeline
        if steps[-1][1].__class__ in [MyKerasClassifier, MyKerasRegressor]:
            fit_params[f'{steps[-1][0]}__with_generator'] = True
            fit_params[f'{steps[-1][0]}__generator'] = aug_obj.datagen
            fit_params[f'{steps[-1][0]}__batch_size'] = aug_obj.batch_size
            pipeline.steps = steps[: aug_index] + steps[aug_index + 1:]

        return pipeline, fit_params

    def _preprocess_x(self, pipeline, X_train, X_val):
        # drop fit_resample step
        transform_steps = [(x[1], x[2]) for x in pipeline._iter()]
        # only model
        if len(transform_steps) == 1:
            return X_val
        else:
            pre_pipeline = copy.deepcopy(pipeline)
            # drop model step
            pre_pipeline.steps = transform_steps[:-1]
            pre_pipeline.fit(X_train)
            return pre_pipeline.transform(X_val)

    def _add_eval_to_fit_params(
        self, pipeline, fit_params, X_train, X_val, Y_val
    ):
        steps = pipeline.steps

        if steps[-1][1].__class__ not in [
            MyKerasClassifier, MyKerasRegressor,
            XGBClassifier, XGBRegressor,
            LGBMClassifier, LGBMRegressor,
            TabNetClassifier, TabNetRegressor,
        ]:
            return fit_params

        eval_X_val = self._preprocess_x(pipeline, X_train, X_val)
        if steps[-1][1].__class__ in [MyKerasClassifier, MyKerasRegressor]:
            fit_params[f'{steps[-1][0]}__validation_data'] = \
                (eval_X_val, Y_val)
        elif steps[-1][1].__class__ in [
            XGBClassifier, XGBRegressor,
            LGBMClassifier, LGBMRegressor,
            TabNetClassifier, TabNetRegressor,
        ]:
            fit_params[f'{steps[-1][0]}__eval_set'] = \
                [(eval_X_val, Y_val)]
        return fit_params

    def _add_tabnet_unsupervised_model_to_fit_params(
        self, pipeline, fit_params, X_train, X_val
    ):
        steps = pipeline.steps

        if steps[-1][1].__class__ not in [
            TabNetClassifier, TabNetRegressor,
        ]:
            return fit_params

        eval_X_train = self._preprocess_x(pipeline, X_train, X_train)
        eval_X_val = self._preprocess_x(pipeline, X_train, X_val)

        logger.info('tabnet unsupervised pre-training')
        _uns_fit_params = {}
        for k, v in fit_params.items():
            if f'{steps[-1][0]}__' not in k:
                continue
            if 'from_unsupervised' in k or 'eval' in k:
                continue

            _uns_fit_params[k.replace(f'{steps[-1][0]}__', '')] = v
        unsupervised_model = TabNetPretrainer()
        unsupervised_model.fit(
            X_train=eval_X_train, eval_set=[eval_X_val], **_uns_fit_params)

        fit_params[f'{steps[-1][0]}__from_unsupervised'] = unsupervised_model
        return fit_params

    def _get_feature_importances(self, pipeline):
        _estimator = pipeline.steps[-1][1]
        if not hasattr(_estimator, 'feature_importances_'):
            return None

        feature_importances = pd.DataFrame(
            data=[_estimator.feature_importances_],
            columns=self.feature_columns)
        feature_importances = feature_importances.iloc[
            :, np.argsort(feature_importances.to_numpy()[0])[::-1]]
        return feature_importances / np.sum(feature_importances.to_numpy())

    def _get_permutation_importances(self, pipeline, X_train, Y_train):
        if not self.configs['fit'].get('permutation'):
            return None

        if X_train.ndim > 2:
            return None

        if len(self.feature_columns) > 50:
            logger.warning('COLUMNS IS TOO LARGE, THEN NO PERMUTATION')
            return None

        _estimator = pipeline.steps[-1][1]
        if not hasattr(_estimator, 'score'):
            logger.warning('NO SCORE METHOD, THEN NO PERMUTATION')
            return None

        perm = PermutationImportance(_estimator, random_state=42).fit(
            self.toarray_like(X_train), Y_train)
        return eli5.explain_weights_df(
            perm, feature_names=self.feature_columns)

    def calc_cv_scores_pipelines(
        self, pipeline, X_train, Y_train,
        scorer, cv, fit_params, with_importances=False
    ):
        scores = []
        pipelines = []
        if cv == 1:
            logger.warning('TRAIN = VAL SINCE CV=1')
            indexes = [
                [range(X_train.shape[0]), range(X_train.shape[0])]
            ]
        else:
            indexes = cv.split(X_train, Y_train)

        X_train_for_fit, Y_train_for_fit = \
            self._trans_xy_for_fit(pipeline, X_train, Y_train)
        for i, (train_index, val_index) in enumerate(indexes):
            # cvごとに依存させないため、deepcopy
            _pipeline = copy.deepcopy(pipeline)

            # update fit_params
            _pipeline, fit_params = \
                self._add_augmentor_to_fit_params(_pipeline, fit_params)
            fit_params = self._add_eval_to_fit_params(
                _pipeline, fit_params,
                X_train_for_fit[train_index],
                X_train_for_fit[val_index],
                Y_train_for_fit[val_index])
            fit_params = self._add_tabnet_unsupervised_model_to_fit_params(
                _pipeline, fit_params,
                X_train_for_fit[train_index],
                X_train_for_fit[val_index])

            # fit
            _pipeline.fit(
                X_train_for_fit[train_index],
                Y_train_for_fit[train_index],
                **fit_params)
            pipelines.append(_pipeline)
            scores.append(scorer(
                _pipeline,
                X_train_for_fit[val_index], Y_train[val_index]))

            # importances
            if not with_importances:
                continue

            _feature_importances = self._get_feature_importances(_pipeline)
            if _feature_importances is not None:
                logger.info(f'  feature importances #{i}')
                logger.info(_feature_importances)

            _perm_importances = self._get_permutation_importances(
                _pipeline,
                X_train_for_fit[train_index], Y_train_for_fit[train_index])
            if _perm_importances is not None:
                logger.info(f'  permutation importances #{i}')
                logger.info(_perm_importances)

        logger.info('    scores: %s' % scores)
        logger.info('    score mean: %s' % np.mean(scores))
        logger.info('    score std: %s' % np.std(scores))
        logger.info('    score max: %s' % np.max(scores))
        logger.info('    score min: %s' % np.min(scores))
        return scores, pipelines

    def calc_best_params(
        self,
        base_pipeline, X_train, Y_train,
        params, scorer, train_cv, fit_params,
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
            logger.info('  params: %s' % args)

            pipeline = base_pipeline
            pipeline.set_params(**args)
            pipeline = self.to_second_estimator(
                pipeline, multiclass, undersampling)

            try:
                scores, _ = self.calc_cv_scores_pipelines(
                    pipeline, X_train, Y_train,
                    scorer, train_cv, fit_params)
            except Exception as e:
                logger.warning(e)
                logger.warning('SET SCORE 0')
                scores = [0]

            # データリーク対策のために、holdoutを想定してminを採用
            return -1 * np.min(scores)

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
                    'N_TRIALS IS OVER MAX PATTERN THEN SET WITH MAX')
                n_trials = all_comb_num
            elif n_trials < 0:
                logger.error(f'N_TRIALS SHOULD BE LARGER THAN 0: {n_trials}')
                raise Exception('ILLEGAL VALUE')
        else:
            if n_trials is None:
                logger.error('IF PARAMS HAVE DICT, N_TRIALS SHOULD BE SET')
                raise Exception('ILLEGAL VALUE')
        logger.info(f'n_trials: {n_trials}')

        study = optuna.create_study()
        study.optimize(_objective, n_trials=n_trials)
        best_params = study.best_params
        best_score_min = -1 * study.best_trial.value
        logger.info('best score min: %s' % best_score_min)
        return best_params
