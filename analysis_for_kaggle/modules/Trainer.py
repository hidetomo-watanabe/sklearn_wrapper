import dill
import scipy.sparse as sp
import numpy as np
import pandas as pd
import importlib
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
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
import torch
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from heamy.dataset import Dataset
from heamy.estimator import Classifier, Regressor
from heamy.pipeline import ModelsPipeline
from sklearn.metrics import get_scorer
from keras.utils.np_utils import to_categorical
from IPython.display import display
from logging import getLogger


logger = getLogger('predict').getChild('Trainer')
if 'ConfigReader' not in globals():
    from .ConfigReader import ConfigReader
if 'CommonMethodWrapper' not in globals():
    from .CommonMethodWrapper import CommonMethodWrapper
if 'Outputer' not in globals():
    from .Outputer import Outputer


class SingleTrainer(ConfigReader, CommonMethodWrapper):
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

    @classmethod
    def get_base_model(self, model, create_nn_model=None):
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

    def _calc_cv_scores_models(
        self, model, X_train, Y_train, scorer, cv, fit_params={}
    ):
        if Y_train.ndim > 1 and Y_train.shape[1] > 1:
            tmp_Y_train = np.argmax(Y_train, axis=1)
        else:
            tmp_Y_train = Y_train
        scores = []
        models = []
        if cv == 1:
            indexes = [[range(X_train.shape[0]), range(tmp_Y_train.shape[0])]]
        else:
            indexes = cv.split(X_train, tmp_Y_train)
        for train_index, test_index in indexes:
            tmp_model = model
            tmp_model.fit(
                X_train[train_index], Y_train[train_index],
                **fit_params)
            models.append(tmp_model)
            scores.append(scorer(
                tmp_model, X_train[test_index], tmp_Y_train[test_index]))
        return scores, models

    def _calc_best_params(
        self,
        base_model, X_train, Y_train, params, scorer, cv,
        fit_params={}, n_trials=None, multiclass=None
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
            model = base_model
            args = _get_args(trial, params)
            model.set_params(**args)
            if multiclass:
                model = multiclass(estimator=model)
            try:
                scores, _ = self._calc_cv_scores_models(
                    model, X_train, Y_train, scorer, cv, fit_params)
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
        self.base_model = self.get_base_model(
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
        cv_select = model_config.get('cv_select')
        if not cv_select:
            cv_select = 'nearest_mean'
        self.cv_select = cv_select
        self.n_trials = model_config.get('n_trials')
        fit_params = model_config.get('fit_params')
        if not fit_params:
            fit_params = {}
        if model in ['lgb_clf', 'lgb_reg']:
            fit_params['eval_set'] = [(X_train, Y_train)]
        self.fit_params = fit_params
        params = model_config.get('params')
        if not params:
            params = {}
        self.params = params
        return

    def _fit(self, scorer, cv, X_train, Y_train):
        best_params = self._calc_best_params(
            self.base_model, X_train, Y_train, self.params,
            scorer, cv, self.fit_params, self.n_trials, self.multiclass)
        logger.info('best params: %s' % best_params)
        estimator = self.base_model
        estimator.set_params(**best_params)
        if self.multiclass:
            estimator = self.multiclass(estimator=estimator)
        logger.info(f'get estimator with cv_select: {self.cv_select}')
        if self.cv_select == 'train_all':
            scores, estimators = self._calc_cv_scores_models(
                estimator, X_train, Y_train, scorer,
                cv=1, fit_params=self.fit_params)
        elif self.cv_select in ['nearest_mean', 'all_folds']:
            scores, estimators = self._calc_cv_scores_models(
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

        # Y_train
        if self.model in ['keras_clf', 'torch_clf']:
            if Y_train.ndim > 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()
            else:
                logger.error('NOT IMPLEMENTED MULTI TARGET KERAS, TORCH CLF')
                raise Exception('NOT IMPLEMENTED')
            if self.model == 'keras_clf':
                Y_train = to_categorical(Y_train)
            elif self.model == 'torch_clf':
                Y_train = torch.LongTensor(Y_train)
        elif self.model not in ['keras_reg', 'torch_reg']:
            # for warning
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
                classes = sorted(np.unique(self.Y_train))

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


class EnsembleTrainer(ConfigReader, CommonMethodWrapper):
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
                estimator=SingleTrainer.get_base_model(
                    ensemble_config['model']).__class__)
        elif self.configs['fit']['train_mode'] == 'reg':
            stacker = Regressor(
                dataset=stack_dataset,
                estimator=SingleTrainer.get_base_model(
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


class Trainer(ConfigReader, CommonMethodWrapper):
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
    def get_cv_from_json(self, cv_config, train_mode):
        if not cv_config:
            if train_mode == 'reg':
                model = KFold(
                    n_splits=3, shuffle=True, random_state=42)
            elif train_mode == 'clf':
                model = StratifiedKFold(
                    n_splits=3, shuffle=True, random_state=42)
            cv = model
            return cv

        fold = cv_config['fold']
        num = cv_config['num']
        if num == 1:
            cv = 1
            return cv

        if fold == 'timeseries':
            model = TimeSeriesSplit(n_splits=num)
        elif fold == 'k':
            model = KFold(
                n_splits=num, shuffle=True, random_state=42)
        elif fold == 'stratifiedk':
            model = StratifiedKFold(
                n_splits=num, shuffle=True, random_state=42)
        else:
            logger.error(f'NOT IMPLEMENTED CV: {fold}')
            raise Exception('NOT IMPLEMENTED')
        cv = model
        return cv

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
            'cv': self.cv,
            'scorer': self.scorer,
            'classes': self.classes,
            'single_estimators': self.single_estimators,
            'estimator': self.estimator,
        }
        return output

    def calc_estimator_data(self):
        # configs
        model_configs = self.configs['fit']['single_model_configs']
        self.cv = self.get_cv_from_json(
            self.configs['fit'].get('cv'), self.configs['fit']['train_mode'])
        logger.info(f'cv: {self.cv}')
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
                config, self.scorer, self.cv, nn_func=myfunc)
            single_scores.append(_score)
            modelname = config.get('modelname')
            if not modelname:
                modelname = f'tmp_model'
            self.single_estimators.append(
                (f'{i}_{modelname}', _estimator))

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
                    self.classes = sorted(np.unique(self.Y_train))
        return self.estimator

    def write_estimator_data(self):
        modelname = \
            self.configs['fit']['ensemble_model_config'].get('modelname')
        if not modelname:
            modelname = 'tmp_model'
        if len(self.single_estimators) == 1:
            targets = self.single_estimators
        else:
            targets = self.single_estimators + [
                (modelname, self.estimator)
            ]
        for modelname, estimator in targets:
            output_path = self.configs['data']['output_dir']
            if estimator.__class__ in [
                NeuralNetClassifier, NeuralNetRegressor
            ]:
                logger.warning('NOT IMPLEMENTED TORCH MODEL SAVE')
            elif hasattr(estimator, 'save'):
                estimator.save(
                    '%s/%s.pickle' % (output_path, modelname))
            else:
                with open(
                    '%s/%s.pickle' % (output_path, modelname), 'wb'
                ) as f:
                    dill.dump(estimator, f)
        return modelname
