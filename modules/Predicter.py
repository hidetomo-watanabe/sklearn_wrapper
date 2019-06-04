import sys
import os
import math
import pickle
import numpy as np
import pandas as pd
import importlib
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, space_eval, Trials, STATUS_OK
from sklearn.linear_model import LogisticRegression, LinearRegression
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
from keras.engine.sequential import Sequential
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from heamy.dataset import Dataset
from heamy.estimator import Classifier, Regressor
from heamy.pipeline import ModelsPipeline, PipeApply
from sklearn.metrics import get_scorer
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display
from logging import getLogger


logger = getLogger('predict').getChild('Predicter')
try:
    from .ConfigReader import ConfigReader
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')


class Predicter(ConfigReader):
    def __init__(
        self,
        feature_columns, test_ids,
        X_train, Y_train, X_test,
        y_scaler=None, kernel=False
    ):
        self.feature_columns = feature_columns
        self.test_ids = test_ids
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.y_scaler = y_scaler
        self.kernel = kernel
        self.BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
        if self.kernel:
            self.OUTPUT_PATH = '.'
        else:
            self.OUTPUT_PATH = '%s/outputs' % self.BASE_PATH
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

    def _calc_best_params(
        self,
        base_model, X_train, Y_train, params, scorer,
        cv=3, n_jobs=-1, fit_params={}, max_evals=None, multiclass=None
    ):
        def _hyperopt_objective(args):
            model = base_model
            model.set_params(**args)
            if multiclass:
                model = multiclass(estimator=model, n_jobs=n_jobs)
            scores = cross_val_score(
                model, X_train, Y_train,
                scoring=scorer, cv=cv, n_jobs=n_jobs, fit_params=fit_params)
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            logger.debug('  params: %s' % args)
            logger.debug('  score mean: %s' % score_mean)
            logger.debug('  score std: %s' % score_std)
            return {'loss': -1 * score_mean, 'status': STATUS_OK}

        params_space = {}
        all_comb_num = 0
        for key, val in params.items():
            params_space[key] = hp.choice(key, val)
            if all_comb_num == 0:
                all_comb_num = 1
            all_comb_num *= len(val)
        if not max_evals:
            max_evals = all_comb_num

        # no search
        if max_evals == 0:
            return {}

        trials_obj = Trials()
        best_params = fmin(
            fn=_hyperopt_objective, space=params_space,
            algo=tpe.suggest, max_evals=max_evals, trials=trials_obj)
        best_params = space_eval(params_space, best_params)
        best_score_mean = -1 * min(
            [item['result']['loss'] for item in trials_obj.trials])
        logger.info('best score mean: %s' % best_score_mean)
        return best_params

    def calc_single_model(
        self,
        scorer, model_config,
        keras_build_func=None, X_train=None, Y_train=None
    ):
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
            if multiclass == 'onevsone':
                multiclass = OneVsOneClassifier
            elif multiclass == 'onevsrest':
                multiclass = OneVsRestClassifier
            else:
                logger.error(
                    f'NOT IMPLEMENTED MULTICLASS: {multiclass}')
                raise Exception('NOT IMPLEMENTED')
        cv = model_config.get('cv')
        if not cv:
            cv = 3
        logger.info('search with cv=%d' % cv)
        n_jobs = model_config.get('n_jobs')
        if not n_jobs:
            n_jobs = -1
        max_evals = model_config.get('max_evals')
        fit_params = model_config.get('fit_params')
        if not fit_params:
            fit_params = {}
        if len(fit_params.keys()) > 0:
            fit_params['eval_set'] = [[X_train, Y_train]]
        params = model_config.get('params')
        if not params:
            params = {}
        if self.configs['fit'].get('time_series'):
            indexes = np.arange(len(Y_train))
            cv_splits = []
            cv_unit = int(len(indexes) / (cv + 1))
            for i in range(cv):
                if i == (cv - 1):
                    end = len(indexes)
                else:
                    end = (i + 2) * cv_unit
                cv_splits.append(
                    (indexes[i * cv_unit: (i + 1) * cv_unit],
                        indexes[(i + 1) * cv_unit: end]))
            cv = cv_splits
        else:
            if self.configs['fit']['train_mode'] == 'reg':
                cv = KFold(
                    n_splits=cv, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=42)

        # for warning
        if model not in ['keras_clf', 'keras_reg']:
            if Y_train.ndim > 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()

        # fit
        best_params = self._calc_best_params(
            base_model, X_train, Y_train, params,
            scorer, cv, n_jobs, fit_params, max_evals, multiclass)
        logger.info('best params: %s' % best_params)
        estimator = base_model
        estimator.set_params(**best_params)
        if multiclass:
            estimator = multiclass(estimator=estimator, n_jobs=n_jobs)
        estimator.fit(X_train, Y_train, **fit_params)
        logger.info('estimator: %s' % estimator)

        # feature_importances
        if hasattr(estimator, 'feature_importances_'):
            feature_importances = pd.DataFrame(
                data=[estimator.feature_importances_],
                columns=self.feature_columns)
            feature_importances = feature_importances.ix[
                :, np.argsort(feature_importances.values[0])[::-1]]
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
        return estimator

    def get_estimator_data(self):
        output = {
            'scorer': self.scorer,
            'single_estimators': self.single_estimators,
            'estimator': self.estimator,
        }
        return output

    def calc_ensemble_model(self):
        def _get_scorer_from_config():
            scoring = self.configs['fit']['scoring']
            logger.info('scoring: %s' % scoring)
            if scoring == 'my_scorer':
                if not self.kernel:
                    myfunc = importlib.import_module(
                        'modules.myfuncs.%s'
                        % self.configs['fit'].get('myfunc'))
                method_name = 'get_my_scorer'
                if not self.kernel:
                    method_name = 'myfunc.%s' % method_name
                self.scorer = eval(method_name)()
            else:
                self.scorer = get_scorer(scoring)
            return self.scorer

        def _get_stacker(pipeline, ensemble_config):
            # weighted_average
            if ensemble_config['mode'] == 'weighted':
                weights = pipeline.find_weights(self.scorer._score_func)
                stacker = pipeline.weight(weights)
                return stacker

            # stacking, blending
            if ensemble_config['mode'] == 'stacking':
                stack_dataset = pipeline.stack(
                    k=ensemble_config['k'], seed=ensemble_config['seed'])
            elif ensemble_config['mode'] == 'blending':
                stack_dataset = pipeline.blend(
                    proportion=ensemble_config['proportion'],
                    seed=ensemble_config['seed'])
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
            return stacker

        # configs
        model_configs = self.configs['fit']['single_models']
        self.scorer = _get_scorer_from_config()
        myfunc = self.configs['fit'].get('myfunc')
        self.classes = None
        logger.info('X train shape: %s' % str(self.X_train.shape))
        logger.info('Y train shape: %s' % str(self.Y_train.shape))

        # single
        if len(model_configs) == 1:
            logger.warn('NO ENSEMBLE')
            self.estimator = self.calc_single_model(
                self.scorer, model_configs[0], keras_build_func=myfunc)
            self.single_estimators = [(model_configs[0], self.estimator)]
            if self.configs['fit']['train_mode'] == 'clf':
                if hasattr(self.estimator, 'classes_'):
                    self.classes = self.estimator.classes_
                else:
                    self.classes = sorted(np.unique(self.Y_train))
            return self.estimator

        # ensemble
        logger.info('single fit in ensemble')
        models = []
        self.single_estimators = []
        # for warning
        if self.Y_train.ndim > 1 and self.Y_train.shape[1] == 1:
            dataset = Dataset(self.X_train, self.Y_train.ravel(), self.X_test)
        else:
            dataset = Dataset(self.X_train, self.Y_train, self.X_test)

        # single fit
        for model_config in model_configs:
            single_estimator = self.calc_single_model(
                self.scorer, model_config, keras_build_func=myfunc)
            self.single_estimators.append((model_config, single_estimator))
            modelname = model_config.get('modelname')
            if not modelname:
                modelname = 'tmp_model'
            # clf
            if self.configs['fit']['train_mode'] == 'clf':
                if self.classes is None:
                    if hasattr(single_estimator, 'classes_'):
                        self.classes = single_estimator.classes_
                    else:
                        self.classes = sorted(np.unique(self.Y_train))
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

        # pipeline
        ensemble_config = self.configs['fit']['ensemble']
        pipeline = ModelsPipeline(*models)

        # stacker
        if ensemble_config['mode'] == 'weighted' \
                and self.configs['fit']['train_mode'] == 'clf':
            logger.error(
                'NOT IMPLEMENTED CLASSIFICATION AND WEIGHTED AVERAGE')
            raise Exception('NOT IMPLEMENTED')
        stacker = _get_stacker(pipeline, ensemble_config)

        # validate
        stacker.probability = False
        logger.info('ensemble validation')
        stacker.validate(
            k=ensemble_config['k'], scorer=self.scorer._score_func)

        self.estimator = stacker
        return self.estimator

    def write_estimator_data(self):
        modelname = self.configs['fit']['ensemble'].get('modelname')
        if not modelname:
            modelname = 'tmp_model'
        if len(self.single_estimators) == 1:
            targets = self.single_estimators
        else:
            targets = self.single_estimators + [
                ({'modelname': modelname}, self.estimator)
            ]
        for config, estimator in targets:
            modelname = config.get('modelname')
            if not modelname:
                modelname = 'tmp_model'
            if hasattr(estimator, 'save'):
                estimator.save(
                    '%s/%s.pickle' % (self.OUTPUT_PATH, modelname))
            else:
                with open(
                    '%s/%s.pickle' % (self.OUTPUT_PATH, modelname), 'wb'
                ) as f:
                    pickle.dump(estimator, f)
        return

    def get_predict_data(self):
        output = {
            'Y_pred': self.Y_pred,
            'Y_pred_proba': self.Y_pred_proba,
            'Y_train_pred': self.Y_train_pred,
            'Y_pred_df': self.Y_pred_df,
            'Y_pred_proba_df': self.Y_pred_proba_df,
        }
        return output

    def predict_y(self):
        self.Y_pred = None
        self.Y_pred_proba = None
        self.Y_train_pred = None
        # clf
        if self.configs['fit']['train_mode'] == 'clf':
            # keras clf
            if self.estimator.__class__ in [Sequential]:
                self.Y_pred = self.estimator.predict_classes(self.X_test)
                self.Y_pred_proba = self.estimator.predict(self.X_test)
                self.Y_train_pred = self.estimator.predict_classes(
                    self.X_train)
            # ensemble clf
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
                dataset = Dataset(self.X_train, self.Y_train, self.X_train)
                self.estimator.dataset = dataset
                self.Y_train_pred = self.estimator.predict()
            # single clf
            else:
                self.Y_pred = self.estimator.predict(self.X_test)
                self.Y_train_pred = self.estimator.predict(self.X_train)
                if hasattr(self.estimator, 'predict_proba'):
                    self.Y_pred_proba = self.estimator.predict_proba(
                        self.X_test)
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
                logger.warn('NO Y_train_pred')
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
        return self.Y_pred, self.Y_pred_proba, self.Y_train_pred

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
            self.Y_pred_proba_df = pd.DataFrame(
                data=self.test_ids, columns=[self.id_col])
            if len(self.pred_cols) == 1:
                proba_columns = map(
                    lambda x: '%s_%s' % (self.pred_cols[0], str(x)),
                    self.classes)
            else:
                proba_columns = self.pred_cols
            self.Y_pred_proba_df = pd.merge(
                self.Y_pred_proba_df,
                pd.DataFrame(
                    data=self.Y_pred_proba,
                    columns=proba_columns),
                left_index=True, right_index=True)

        # post
        fit_post = self.configs.get('post')
        if fit_post:
            if fit_post['myfunc']:
                if not self.kernel:
                    sys.path.append(self.BASE_PATH)
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
        modelname = self.configs['fit']['ensemble'].get('modelname')
        if not modelname:
            modelname = 'tmp_model'
        filename = '%s.csv' % modelname
        if isinstance(self.Y_pred_df, pd.DataFrame):
            self.Y_pred_df.to_csv(
                '%s/%s' % (self.OUTPUT_PATH, filename), index=False)
        if isinstance(self.Y_pred_proba_df, pd.DataFrame):
            self.Y_pred_proba_df.to_csv(
                '%s/proba_%s' % (self.OUTPUT_PATH, filename),
                index=False)
        return filename
