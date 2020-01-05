import pickle
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
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from heamy.dataset import Dataset
from heamy.estimator import Classifier, Regressor
from heamy.pipeline import ModelsPipeline
from sklearn.metrics import get_scorer
from keras.utils.np_utils import to_categorical
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display
from logging import getLogger


logger = getLogger('predict').getChild('Trainer')
try:
    from .ConfigReader import ConfigReader
except ImportError:
    logger.warning('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')
try:
    from .Outputer import Outputer
except ImportError:
    logger.warning('IN FOR KERNEL SCRIPT, Outputer import IS SKIPPED')


class Trainer(ConfigReader):
    def __init__(
        self,
        feature_columns, test_ids,
        X_train, Y_train, X_test,
        kernel=False
    ):
        self.feature_columns = feature_columns
        self.test_ids = test_ids
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.kernel = kernel
        self.configs = {}
        # kerasのスコープ対策として、インスタンス作成時に読み込み
        # keras使う時しか使わないので、evalで定義してエラー回避
        if self.kernel:
            self.create_keras_model = eval('create_keras_model')

    def _get_base_model(self, model, keras_build_func=None):
        if model in ['keras_clf', 'keras_reg']:
            if self.kernel:
                create_keras_model = self.create_keras_model
            else:
                myfunc = importlib.import_module(
                    'modules.myfuncs.%s' % keras_build_func)
                create_keras_model = myfunc.create_keras_model

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
            return KerasClassifier(build_fn=create_keras_model)
        elif model == 'keras_reg':
            return KerasRegressor(build_fn=create_keras_model)
        else:
            logger.error('NOT IMPLEMENTED BASE MODEL: %s' % model)
            raise Exception('NOT IMPLEMENTED')

    def _get_cv_scores_models(
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
        n_jobs=-1, fit_params={}, n_trials=None, multiclass=None
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
                model = multiclass(estimator=model, n_jobs=n_jobs)
            try:
                scores, _ = self._get_cv_scores_models(
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
            if all_comb_num == 0:
                all_comb_num = 1
            all_comb_num *= len(val)
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

        if n_trials:
            n_trials = min(n_trials, all_comb_num)
        elif n_trials != 0:
            n_trials = all_comb_num
        if n_trials < 0:
            logger.error(f'N_TRIALS SHOULD BE LARGER THAN 0: {n_trials}')
            raise Exception('ILLEGAL VALUE')
        logger.info(f'n_trials: {n_trials}')

        study = optuna.create_study()
        study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)
        best_params = study.best_params
        best_score_mean = -1 * study.best_trial.value
        logger.info('best score mean: %s' % best_score_mean)
        return best_params

    def _check_importances(self, model, estimator, X_train, Y_train):
        # feature_importances
        if hasattr(estimator, 'feature_importances_'):
            feature_importances = pd.DataFrame(
                data=[estimator.feature_importances_],
                columns=self.feature_columns)
            feature_importances = feature_importances.iloc[
                :, np.argsort(feature_importances.to_numpy()[0])[::-1]]
            logger.info('feature importances:')
            display(feature_importances)
            logger.info('feature importances /sum:')
            display(
                feature_importances / np.sum(
                    estimator.feature_importances_))

        # permutation importance
        if self.configs['fit'].get('permutation'):
            if model not in ['keras_clf', 'keras_reg']:
                perm = PermutationImportance(
                    estimator, random_state=42).fit(
                    X_train, Y_train)
                logger.info('permutation importance:')
                display(
                    eli5.explain_weights_df(
                        perm, feature_names=self.feature_columns))
        return

    def get_estimator_data(self):
        output = {
            'cv': self.cv,
            'scorer': self.scorer,
            'classes': self.classes,
            'single_estimators': self.single_estimators,
            'estimator': self.estimator,
        }
        return output

    def calc_estimator(self):
        def _get_cv_from_config():
            cv_config = self.configs['fit'].get('cv')
            if not cv_config:
                if self.configs['fit']['train_mode'] == 'reg':
                    model = KFold(
                        n_splits=3, shuffle=True, random_state=42)
                elif self.configs['fit']['train_mode'] == 'clf':
                    model = StratifiedKFold(
                        n_splits=3, shuffle=True, random_state=42)
                self.cv = model
                return self.cv

            fold = cv_config['fold']
            num = cv_config['num']
            if num == 1:
                self.cv = 1
                return self.cv

            logger.info('search with cv: fold=%s, num=%d' % (fold, num))
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
            self.cv = model
            return self.cv

        # configs
        model_configs = self.configs['fit']['single_models']
        cv = _get_cv_from_config()
        n_jobs = self.configs['fit'].get('n_jobs')
        if not n_jobs:
            n_jobs = -1
        logger.info('scoring: %s' % self.configs['fit']['scoring'])
        self.scorer = self._get_scorer_from_string(
            self.configs['fit']['scoring'])
        myfunc = self.configs['fit'].get('myfunc')
        self.classes = None

        # single
        logger.info('single fit')
        single_scores = []
        self.single_estimators = []
        for i, config in enumerate(model_configs):
            _scores, _estimators = self.calc_single_estimators(
                self.scorer, config, cv, n_jobs,
                keras_build_func=myfunc)
            single_scores.extend(_scores)
            modelname = config.get('modelname')
            if not modelname:
                modelname = f'tmp_model_{i}'
            for j, _estimator in enumerate(_estimators):
                self.single_estimators.append(
                    (f'{modelname}_{j}', _estimator))

        # ensemble
        if len(self.single_estimators) == 1:
            logger.info('no ensemble')
            self.estimator = self.single_estimators[0][1]
        else:
            self.estimator = self.calc_ensemble_estimator(
                self.single_estimators, single_scores, n_jobs)

        # classes
        if self.configs['fit']['train_mode'] == 'clf':
            for _, single_estimator in self.single_estimators:
                if isinstance(self.classes, pd.DataFrame):
                    continue
                if hasattr(single_estimator, 'classes_'):
                    self.classes = single_estimator.classes_
                else:
                    self.classes = sorted(np.unique(self.Y_train))
        return self.estimator

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

    def _calc_pseudo_label_data(
        self, X_train, Y_train, estimator, classes, threshold
    ):
        outputer_obj = Outputer(
            [], [], X_train, Y_train, self.X_test,
            [], None, [], [], estimator)
        outputer_obj.configs = self.configs
        _, Y_pred_proba, _ = outputer_obj.predict_y()

        data_indexes, label_indexes = np.where(Y_pred_proba > threshold)
        pseudo_X_train = self.X_test[data_indexes]
        pseudo_Y_train = []
        for label_index in label_indexes:
            pseudo_Y_train.append(classes[label_index])
        pseudo_Y_train = np.array(pseudo_Y_train)
        return pseudo_X_train, pseudo_Y_train

    def calc_single_estimators(
        self,
        scorer, model_config, cv=KFold(), n_jobs=-1,
        keras_build_func=None, X_train=None, Y_train=None
    ):
        def _fit(X_train, Y_train):
            best_params = self._calc_best_params(
                base_model, X_train, Y_train, params,
                scorer, cv, n_jobs, fit_params, n_trials, multiclass)
            logger.info('best params: %s' % best_params)
            estimator = base_model
            estimator.set_params(**best_params)
            if multiclass:
                estimator = multiclass(estimator=estimator, n_jobs=n_jobs)
            logger.info(f'get estimator with cv_select: {cv_select}')
            if cv_select == 'train_all':
                scores, estimators = self._get_cv_scores_models(
                    estimator, X_train, Y_train, scorer,
                    cv=1, fit_params=fit_params)
            elif cv_select in ['nearest_mean', 'all_folds']:
                scores, estimators = self._get_cv_scores_models(
                    estimator, X_train, Y_train, scorer,
                    cv=cv, fit_params=fit_params)
                logger.info(f'cv model scores mean: {np.mean(scores)}')
                logger.info(f'cv model scores std: {np.std(scores)}')
                if cv_select == 'nearest_mean':
                    nearest_index \
                        = np.abs(np.array(scores) - np.mean(scores)).argmin()
                    scores = scores[nearest_index: nearest_index + 1]
                    estimators = estimators[nearest_index: nearest_index + 1]
            else:
                logger.error(f'NOT IMPLEMENTED CV SELECT: {cv_select}')
                raise Exception('NOT IMPLEMENTED')
            return scores, estimators

        if isinstance(scorer, str):
            scorer = self._get_scorer_from_string(scorer)
        if not isinstance(X_train, np.ndarray):
            X_train = self.X_train
        if not isinstance(Y_train, np.ndarray):
            Y_train = self.Y_train

        # params
        model = model_config['model']
        logger.info('model: %s' % model)
        modelname = model_config.get('modelname')
        if modelname:
            logger.info('modelname: %s' % modelname)
        base_model = self._get_base_model(model, keras_build_func)
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
        cv_select = model_config.get('cv_select')
        if not cv_select:
            cv_select = 'nearest_mean'
        n_trials = model_config.get('n_trials')
        fit_params = model_config.get('fit_params')
        if not fit_params:
            fit_params = {}
        params = model_config.get('params')
        if not params:
            params = {}

        # Y_train
        if model in ['keras_clf']:
            Y_train = to_categorical(Y_train)
        elif model not in ['keras_reg']:
            # for warning
            if Y_train.ndim > 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()

        # fit
        logger.info('fit')
        scores, estimators = _fit(X_train, Y_train)
        logger.info(f'scores: {scores}')
        logger.info(f'estimators: {estimators}')
        # pseudo
        pseudo_config = model_config.get('pseudo')
        if pseudo_config:
            logger.info('refit with pseudo labeling')
            if self.configs['fit']['train_mode'] == 'reg':
                logger.error('NOT IMPLEMENTED PSEUDO LABELING WITH REGRESSION')
                raise Exception('NOT IMPLEMENTED')
            if cv_select == 'all_folds':
                logger.error('NOT IMPLEMENTED PSEUDO LABELING WITH ALL FOLDS')
                raise Exception('NOT IMPLEMENTED')

            estimator = estimators[0]
            threshold = pseudo_config.get('threshold')
            if not threshold and int(threshold) != 0:
                threshold = 0.8
            if hasattr(estimator, 'classes_'):
                classes = estimator.classes_
            else:
                classes = sorted(np.unique(self.Y_train))
            pseudo_X_train, pseudo_Y_train = self._calc_pseudo_label_data(
                X_train, Y_train, estimator, classes, threshold)
            new_X_train = sp.vstack((X_train, pseudo_X_train), format='csr')
            new_Y_train = np.concatenate([Y_train, pseudo_Y_train])
            logger.info(
                'with threshold %s, train data added %s => %s'
                % (threshold, len(Y_train), len(new_Y_train)))
            scores, estimators = _fit(new_X_train, new_Y_train)
            logger.info(f'scores: {scores}')
            logger.info(f'estimators: {estimators}')

        # importances
        if cv_select == 'all_folds':
            logger.warning('CHECK IMPORTANCE FOR ONLY FIRST ESTIMATOR')
        self._check_importances(model, estimators[0], X_train, Y_train)
        return scores, estimators

    def _get_voter(self, mode, estimators, weights=None, n_jobs=-1):
        if self.configs['fit']['train_mode'] == 'clf':
            if mode == 'average':
                voting = 'soft'
            elif mode == 'vote':
                voting = 'hard'
            voter = VotingClassifier(
                estimators=estimators, voting=voting,
                weights=weights, n_jobs=n_jobs)
        elif self.configs['fit']['train_mode'] == 'reg':
            if mode == 'average':
                voter = VotingRegressor(
                    estimators=estimators, weights=weights, n_jobs=n_jobs)
        return voter

    def _get_pipeline(self, single_estimators):
        # for warning
        if self.Y_train.ndim > 1 and self.Y_train.shape[1] == 1:
            dataset = Dataset(
                self.X_train.toarray(), self.Y_train.ravel(),
                self.X_test.toarray())
        else:
            dataset = Dataset(
                self.X_train.toarray(), self.Y_train, self.X_test.toarray())
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
        # weighted_average
        if ensemble_config['mode'] == 'weighted':
            weights = pipeline.find_weights(self.scorer._score_func)
            stacker = pipeline.weight(weights)
            return stacker

        # stacking, blending
        if ensemble_config['mode'] == 'stacking':
            stack_dataset = pipeline.stack(
                k=ensemble_config['k'], seed=42)
        elif ensemble_config['mode'] == 'blending':
            stack_dataset = pipeline.blend(
                proportion=ensemble_config['proportion'], seed=42)
        if self.configs['fit']['train_mode'] == 'clf':
            stacker = Classifier(
                dataset=stack_dataset,
                estimator=self._get_base_model(
                    ensemble_config['model']).__class__)
        elif self.configs['fit']['train_mode'] == 'reg':
            stacker = Regressor(
                dataset=stack_dataset,
                estimator=self._get_base_model(
                    ensemble_config['model']).__class__)
        stacker.use_cache = False
        # default predict
        stacker.probability = False
        return stacker

    def calc_ensemble_estimator(
        self, single_estimators, single_scores, n_jobs=-1,
        X_train=None, Y_train=None
    ):
        if not isinstance(X_train, np.ndarray):
            X_train = self.X_train
        if not isinstance(Y_train, np.ndarray):
            Y_train = self.Y_train

        ensemble_config = self.configs['fit']['ensemble']
        logger.info('ensemble fit: %s' % ensemble_config['mode'])
        if ensemble_config['mode'] in ['average', 'vote']:
            if ensemble_config['mode'] == 'vote' \
                    and self.configs['fit']['train_mode'] == 'reg':
                logger.error(
                    'NOT IMPLEMENTED REGRESSION AND VOTE')
                raise Exception('NOT IMPLEMENTED')

            single_scores = np.array(single_scores)
            weights = single_scores / np.sum(single_scores)
            logger.info(f'weights: {weights}')
            voter = self._get_voter(
                ensemble_config['mode'], single_estimators, weights, n_jobs)
            voter.fit(X_train, Y_train.ravel())
            estimator = voter
        elif ensemble_config['mode'] in ['weighted', 'stacking', 'blending']:
            if ensemble_config['mode'] == 'weighted' \
                    and self.configs['fit']['train_mode'] == 'clf':
                logger.error(
                    'NOT IMPLEMENTED CLASSIFICATION AND WEIGHTED AVERAGE')
                raise Exception('NOT IMPLEMENTED')

            pipeline = self._get_pipeline(single_estimators)
            stacker = self._get_stacker(pipeline, ensemble_config)
            stacker.validate(
                k=ensemble_config['k'], scorer=self.scorer._score_func)
            estimator = stacker
        else:
            logger.error(
                'NOT IMPLEMENTED ENSEMBLE MODE: %s' % ensemble_config['mode'])
            raise Exception('NOT IMPLEMENTED')
        return estimator

    def write_estimator_data(self):
        modelname = self.configs['fit']['ensemble'].get('modelname')
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
            if hasattr(estimator, 'save'):
                estimator.save(
                    '%s/%s.pickle' % (output_path, modelname))
            else:
                with open(
                    '%s/%s.pickle' % (output_path, modelname), 'wb'
                ) as f:
                    pickle.dump(estimator, f)
        return modelname
