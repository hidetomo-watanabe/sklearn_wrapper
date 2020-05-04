import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from rgf.sklearn import RGFClassifier, RGFRegressor
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from skorch import NeuralNetClassifier, NeuralNetRegressor
from torch import cuda
from keras.utils.np_utils import to_categorical
from logging import getLogger


logger = getLogger('predict').getChild('BaseTrainer')
if 'ConfigReader' not in globals():
    from ..ConfigReader import ConfigReader
if 'CommonMethodWrapper' not in globals():
    from ..CommonMethodWrapper import CommonMethodWrapper


class BaseTrainer(ConfigReader, CommonMethodWrapper):
    def __init__(self):
        pass

    @classmethod
    def get_base_estimator(self, model, create_nn_model=None):
        device = 'cuda' if cuda.is_available() else 'cpu'

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
            return KerasClassifier(build_fn=create_nn_model)
        elif model == 'keras_reg':
            return KerasRegressor(build_fn=create_nn_model)
        elif model == 'torch_clf':
            return NeuralNetClassifier(
                module=create_nn_model(), device=device, train_split=None)
        elif model == 'torch_reg':
            return NeuralNetRegressor(
                module=create_nn_model(), device=device, train_split=None)
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
        return estimator

    @classmethod
    def _reshape_y_train_for_keras(self, estimator, Y_train):
        if estimator.__class__ not in [KerasClassifier]:
            return Y_train
        if Y_train.ndim > 2:
            return Y_train
        if Y_train.ndim == 2 and Y_train.shape[1] > 1:
            return Y_train
        logger.info('to_categorical y_train')
        Y_train = to_categorical(Y_train)
        return Y_train

    @classmethod
    def calc_cv_scores_estimators(
        self, estimator, X_train, Y_train, scorer, cv, fit_params={}
    ):
        scores = []
        estimators = []
        if cv == 1:
            indexes = [[range(X_train.shape[0]), range(Y_train.shape[0])]]
        else:
            indexes = cv.split(X_train, Y_train)

        Y_train_for_fit = self._reshape_y_train_for_keras(estimator, Y_train)
        for train_index, test_index in indexes:
            tmp_estimator = estimator
            tmp_estimator.fit(
                X_train[train_index], Y_train_for_fit[train_index],
                **fit_params)
            estimators.append(tmp_estimator)
            scores.append(scorer(
                tmp_estimator, X_train[test_index], Y_train[test_index]))
        return scores, estimators

    @classmethod
    def calc_best_params(
        self,
        base_estimator, X_train, Y_train, params, scorer, cv,
        fit_params={}, n_trials=None, multiclass=None, undersampling=None
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