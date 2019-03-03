import sys
import os
import math
import pickle
import numpy as np
import pandas as pd
import importlib
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
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
        scaler_y=None, kernel=False
    ):
        self.feature_columns = feature_columns
        self.test_ids = test_ids
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.scaler_y = scaler_y
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
        elif model == 'keras_clf':
            return KerasClassifier(build_fn=create_keras_model)
        elif model == 'keras_reg':
            return KerasRegressor(build_fn=create_keras_model)

    def is_ok_with_adversarial_validation(self):
        def _get_adversarial_preds(X_train, X_test, adversarial):
            # create data
            tmp_X_train = X_train[:len(X_test)]
            X_adv = np.concatenate((tmp_X_train, X_test), axis=0)
            target_adv = np.concatenate(
                (np.zeros(len(tmp_X_train)), np.ones(len(X_test))), axis=0)
            # fit
            skf = StratifiedKFold(
                n_splits=adversarial['cv'], shuffle=True, random_state=42)
            gs = GridSearchCV(
                self._get_base_model(adversarial['model']),
                adversarial['params'],
                cv=skf.split(X_adv, target_adv),
                scoring=adversarial['scoring'],
                n_jobs=adversarial['n_jobs'])
            gs.fit(X_adv, target_adv)
            est = gs.best_estimator_
            return est.predict(tmp_X_train), est.predict(X_test)

        def _is_ok_pred_nums(tr0, tr1, te0, te1):
            if tr0 == 0 and te0 == 0:
                return False
            if tr1 == 0 and te1 == 0:
                return False
            if tr1 == 0 and te0 == 0:
                return False
            return True

        X_train = self.X_train
        adversarial = self.configs['data']['adversarial']
        if adversarial:
            logger.info('with adversarial')
            adv_pred_train, adv_pred_test = _get_adversarial_preds(
                X_train, self.X_test, adversarial)
            adv_pred_train_num_0 = len(np.where(adv_pred_train == 0)[0])
            adv_pred_train_num_1 = len(np.where(adv_pred_train == 1)[0])
            adv_pred_test_num_0 = len(np.where(adv_pred_test == 0)[0])
            adv_pred_test_num_1 = len(np.where(adv_pred_test == 1)[0])
            logger.info('pred train num 0: %s' % adv_pred_train_num_0)
            logger.info('pred train num 1: %s' % adv_pred_train_num_1)
            logger.info('pred test num 0: %s' % adv_pred_test_num_0)
            logger.info('pred test num 1: %s' % adv_pred_test_num_1)
            if not _is_ok_pred_nums(
                adv_pred_train_num_0,
                adv_pred_train_num_1,
                adv_pred_test_num_0,
                adv_pred_test_num_1,
            ):
                logger.warn('TRAIN AND TEST MAY BE HAVE DIFFERENT FEATURES')
        else:
            logger.warn('NO DATA VALIDATION')
            return True

    def get_estimator_data(self):
        output = {
            'scorer': self.scorer,
            'single_estimators': self.single_estimators,
            'estimator': self.estimator,
        }
        return output

    def _get_scorer_from_config(self):
        scoring = self.configs['fit']['scoring']
        logger.info('scoring: %s' % scoring)
        if scoring == 'my_scorer':
            if not self.kernel:
                myfunc = importlib.import_module(
                    'modules.myfuncs.%s' % self.configs['fit']['myfunc'])
            method_name = 'get_my_scorer'
            if not self.kernel:
                method_name = 'myfunc.%s' % method_name
            self.scorer = eval(method_name)()
        else:
            self.scorer = get_scorer(scoring)
        return self.scorer

    def calc_ensemble_model(self):
        scorer = self._get_scorer_from_config()
        model_configs = self.configs['fit']['single_models']
        self.classes = None
        logger.info('X train shape: %s' % str(self.X_train.shape))
        logger.info('Y train shape: %s' % str(self.Y_train.shape))

        # single
        if len(model_configs) == 1:
            logger.warn('NO ENSEMBLE')
            self.estimator = self._calc_single_model(
                scorer, model_configs[0], self.configs['fit'].get('myfunc'))
            self.single_estimators = [(model_configs[0], self.estimator)]
            if self.configs['fit']['train_mode'] == 'clf':
                if hasattr(self.estimator, 'classes_'):
                    self.classes = self.estimator.classes_
                else:
                    self.classes = sorted(np.unique(self.Y_train))
            return self.estimator

        # ensemble
        models = []
        self.single_estimators = []
        dataset = Dataset(self.X_train, self.Y_train, self.X_test)
        for model_config in model_configs:
            single_estimator = self._calc_single_model(
                scorer, model_config, self.configs['fit'].get('myfunc'))
            self.single_estimators.append((model_config, single_estimator))
            if self.classes is None \
                    and self.configs['fit']['train_mode'] == 'clf':
                self.classes = single_estimator.classes_
            modelname = model_config['modelname']
            if self.configs['fit']['train_mode'] == 'clf':
                models.append(
                    Classifier(
                        dataset=dataset, estimator=single_estimator.__class__,
                        parameters=single_estimator.get_params(),
                        name=modelname))
            elif self.configs['fit']['train_mode'] == 'reg':
                models.append(
                    Regressor(
                        dataset=dataset, estimator=single_estimator.__class__,
                        parameters=single_estimator.get_params(),
                        name=modelname))
        # pipeline
        ensemble_config = self.configs['fit']['ensemble']
        pipeline = ModelsPipeline(*models)

        def _get_stacker():
            # weighted_average
            if ensemble_config['mode'] == 'weighted':
                weights = pipeline.find_weights(scorer._score_func)
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

        # stacker
        if ensemble_config['mode'] == 'weighted' \
                and self.configs['fit']['train_mode'] == 'clf':
            logger.error(
                'NOT IMPLEMENTED CLASSIFICATION AND WEIGHTED AVERAGE')
            raise Exception('NOT IMPLEMENTED')
        stacker = _get_stacker()

        # validate
        stacker.probability = False
        logger.info('ENSEMBLE VALIDATION:')
        stacker.validate(k=ensemble_config['k'], scorer=scorer._score_func)

        self.estimator = stacker
        return self.estimator

    def _calc_single_model(self, scorer, model_config, myfunc):
        model = model_config['model']
        modelname = model_config['modelname']
        base_model = self._get_base_model(model, myfunc)
        multiclass = model_config.get('multiclass')
        if multiclass:
            if multiclass == 'onevsone':
                base_model = OneVsOneClassifier(base_model)
            elif multiclass == 'onevsrest':
                base_model = OneVsRestClassifier(base_model)
        cv = model_config['cv']
        n_jobs = model_config['n_jobs']
        fit_params = model_config['fit_params']
        params = model_config['params']
        if len(fit_params.keys()) > 0:
            fit_params['eval_set'] = [[self.X_train, self.Y_train]]

        logger.info('model: %s' % model)
        logger.info('modelname: %s' % modelname)
        logger.info('grid search with cv=%d' % cv)
        if multiclass or self.configs['fit']['train_mode'] == 'reg':
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            cv = kf.split(self.X_train, self.Y_train)
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            cv = skf.split(self.X_train, self.Y_train)
        gs = GridSearchCV(
            estimator=base_model, param_grid=params,
            cv=cv, scoring=scorer, n_jobs=n_jobs,
            return_train_score=False, refit=True)
        gs.fit(self.X_train, self.Y_train, **fit_params)
        logger.info('  best params: %s' % gs.best_params_)
        logger.info('  best score of trained grid search: %s' % gs.best_score_)
        logger.info('  score std of trained grid search: %s' %
                    gs.cv_results_['std_test_score'][gs.best_index_])
        if model in ['keras_clf', 'keras_reg']:
            estimator = gs.best_estimator_.model
        else:
            estimator = gs.best_estimator_
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
                perm = PermutationImportance(estimator, random_state=42).fit(
                    self.X_train, self.Y_train)
                logger.info('permutation importance:')
                display(
                    eli5.explain_weights_df(
                        perm, feature_names=self.feature_columns))

        return estimator

    def write_estimator_data(self):
        modelname = self.configs['fit']['ensemble']['modelname']
        if len(self.single_estimators) == 1:
            targets = self.single_estimators
        else:
            targets = self.single_estimators + [
                ({'modelname': modelname}, self.estimator)
            ]
        for config, estimator in targets:
            modelname = config['modelname']
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
            # ss
            self.Y_pred = self.scaler_y.inverse_transform(self.Y_pred)
            if isinstance(self.Y_train_pred, np.ndarray):
                self.Y_train_pred = \
                    self.scaler_y.inverse_transform(self.Y_train_pred)
            else:
                logger.warn('NO Y_train_pred')
            # other
            y_pre = self.configs['fit']['y_pre']
            if y_pre:
                logger.info('inverse translate y_train with %s' % y_pre)
                if y_pre == 'log':
                    self.Y_pred = np.array(list(map(math.exp, self.Y_pred)))
                    self.Y_train_pred = np.array(list(map(
                        math.exp, self.Y_train_pred)))
                else:
                    logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
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
        fit_post = self.configs['fit']['post']
        if fit_post['myfunc']:
            if not self.kernel:
                sys.path.append(self.BASE_PATH)
                myfunc = importlib.import_module(
                    'myfuncs.%s' % fit_post['myfunc'])
        for method_name in fit_post['methods']:
            logger.info('fit post: %s' % method_name)
            if not self.kernel:
                method_name = 'myfunc.%s' % method_name
            self.Y_pred_df, self.Y_pred_proba_df = eval(
                method_name)(self.Y_pred_df, self.Y_pred_proba_df)

        return self.Y_pred_df, self.Y_pred_proba_df

    def write_predict_data(self, filename=None):
        if not filename:
            filename = '%s.csv' % self.configs['fit']['ensemble']['modelname']
        if isinstance(self.Y_pred_df, pd.DataFrame):
            self.Y_pred_df.to_csv(
                '%s/%s' % (self.OUTPUT_PATH, filename), index=False)
        if isinstance(self.Y_pred_proba_df, pd.DataFrame):
            self.Y_pred_proba_df.to_csv(
                '%s/proba_%s' % (self.OUTPUT_PATH, filename),
                index=False)
        return filename
