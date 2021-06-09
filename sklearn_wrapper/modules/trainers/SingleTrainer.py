import importlib
from logging import getLogger

from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np

import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


logger = getLogger('predict').getChild('SingleTrainer')
if 'Flattener' not in globals():
    from ..commons.Flattener import Flattener
if 'Reshaper' not in globals():
    from ..commons.Reshaper import Reshaper
if 'BaseTrainer' not in globals():
    from .BaseTrainer import BaseTrainer
if 'EnsembleTrainer' not in globals():
    from .EnsembleTrainer import EnsembleTrainer
if 'Augmentor' not in globals():
    from .Augmentor import Augmentor
if 'Outputer' not in globals():
    from ..Outputer import Outputer


class SingleTrainer(BaseTrainer):
    def __init__(
        self, X_train, Y_train, X_test, feature_columns, configs, kernel=False
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.feature_columns = feature_columns
        self.configs = configs
        self.kernel = kernel
        # keras, torchのスコープ対策として、インスタンス作成時に読み込み
        # keras, torch使う時しか使わないので、evalで定義してエラー回避
        if self.kernel:
            self.create_nn_model = eval('create_nn_model')

    def _to_pipeline_params(self, pre, params):
        return {f'{pre}__{k}': v for k, v in params.items()}

    def _get_model_params(self, model_config):
        # model
        model = model_config['model']
        logger.info('model: %s' % model)
        self.model = model
        modelname = model_config.get('modelname')
        if modelname:
            logger.info('modelname: %s' % modelname)

        # params
        params = model_config.get('params', {})
        self.params = self._to_pipeline_params(self.model, params)

        # fit_params
        fit_params = model_config.get('fit_params', {})
        if self.model in ['keras_clf', 'keras_reg']:
            fit_params['callbacks'] = []
            if fit_params.get('reduce_lr'):
                fit_params['callbacks'].append(
                    ReduceLROnPlateau(**fit_params['reduce_lr']))
                del fit_params['reduce_lr']
            if fit_params.get('early_stopping'):
                fit_params['callbacks'].append(
                    EarlyStopping(**fit_params['early_stopping']))
                del fit_params['early_stopping']
        self.fit_params = self._to_pipeline_params(self.model, fit_params)

        # cv
        self.cv_select = model_config.get('cv_select', 'min')
        self.n_trials = model_config.get('n_trials')

        # multiclass
        # final estimatorに学習後追加
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

        return

    def _add_sampling_to_pipeline(self, pipeline, model_config, X_train):
        undersampling = model_config.get('undersampling')
        oversampling = model_config.get('oversampling')

        # 形式エラー対策 pre
        if (undersampling and undersampling == 'random') or oversampling:
            pipeline.append(('flattener', Flattener()))

        # pipeline
        undersampling_clf = None
        if undersampling:
            logger.info(f'undersampling: {undersampling}')
            if undersampling == 'random':
                pipeline.append(
                    ('undersampling', RandomUnderSampler(random_state=42)))
            # final estimatorに学習後追加
            elif undersampling == 'bagging':
                undersampling_clf = BalancedBaggingClassifier
            elif undersampling == 'adaboost':
                undersampling_clf = RUSBoostClassifier
            else:
                logger.error(
                    f'NOT IMPLEMENTED UNDERSAMPLING: {undersampling}')
                raise Exception('NOT IMPLEMENTED')

        if oversampling:
            logger.info(f'oversampling: {oversampling}')
            if oversampling == 'random':
                pipeline.append(
                    ('oversampling', RandomOverSampler(random_state=42)))
            elif oversampling == 'smote':
                pipeline.append(
                    ('oversampling', SMOTE(random_state=42)))
            else:
                logger.error(
                    f'NOT IMPLEMENTED OVERSAMPLING: {oversampling}')
                raise Exception('NOT IMPLEMENTED')

        # 形式エラー対策 post
        if (undersampling and undersampling == 'random') or oversampling:
            _is_categorical = (self.model == 'keras_clf')
            pipeline.append(
                ('reshaper', Reshaper(X_train.shape[1:], _is_categorical)))
        return pipeline, undersampling_clf

    def _get_base_pipeline(self, model_config, nn_func, X_train):
        _pipeline = []

        # imputation
        imputation = model_config.get('missing_imputation')
        if imputation:
            logger.info(f'missing_imputation: {imputation}')
            _pipeline.append((
                'missing_imputation',
                SimpleImputer(missing_values=np.nan, strategy=imputation)
            ))

        # x_scaler
        x_scaler = model_config.get('x_scaler')
        if x_scaler:
            logger.info(f'x_scaler: {x_scaler}')
            if x_scaler == 'standard':
                # to-do: 外れ値対策として、1-99%に限定検討
                # winsorize(X_train, limits=[0.01, 0.01]).tolist()
                _x_scaler = StandardScaler(with_mean=False)
            elif x_scaler == 'maxabs':
                _x_scaler = MaxAbsScaler()
            _pipeline.append(('x_scaler', _x_scaler))

        # dimension_reduction
        di_reduction = model_config.get('dimension_reduction')
        if di_reduction:
            n = di_reduction['n']
            model = di_reduction['model']
            if n == 'all':
                n = X_train.shape[1]
            logger.info(
                'dimension_reduction: %s to %s with %s'
                % (X_train.shape[1], n, model))

            if model == 'pca':
                # pca ratioがuniqueでないと、再現性ない場合あり
                _pipeline.append((
                    'dimension_reduction',
                    PCA(n_components=n, random_state=42)
                ))
            elif model == 'svd':
                _pipeline.append((
                    'dimension_reduction',
                    TruncatedSVD(n_components=n, random_state=42)
                ))
            elif model == 'kmeans':
                _pipeline.append((
                    'dimension_reduction',
                    KMeans(n_clusters=n, random_state=42, n_jobs=-1)
                ))
            elif model == 'nmf':
                _pipeline.append((
                    'dimension_reduction',
                    NMF(n_components=n, random_state=42)
                ))
            else:
                logger.error(
                    'NOT IMPLEMENTED DIMENSION REDUCTION MODEL: %s' % model)
                raise Exception('NOT IMPLEMENTED')
            self.feature_columns = list(map(
                lambda x: '%s_%d' % (model, x), range(n)))

        # sampling
        _pipeline, self.undersampling = \
            self._add_sampling_to_pipeline(_pipeline, model_config, X_train)

        # augmentation
        augmentation = model_config.get('augmentation')
        if augmentation:
            logger.info(f'augmentation: {augmentation}')
            _pipeline.append(
                ('augmentation', Augmentor(**augmentation)))

        # model
        create_nn_model = None
        if self.model in ['keras_clf', 'keras_reg', 'torch_clf', 'torch_reg']:
            if self.kernel:
                create_nn_model = self.create_nn_model
            else:
                myfunc = importlib.import_module(
                    'modules.myfuncs.%s' % nn_func)
                create_nn_model = myfunc.create_nn_model
        _pipeline.append((
            self.model,
            self.get_base_estimator(
                self.model, create_nn_model=create_nn_model)
        ))

        return Pipeline(_pipeline)

    def _fit(self, scorer, train_cv, val_cv, X_train, Y_train):
        # for param tuning, use train_cv
        best_params = self.calc_best_params(
            self.base_pipeline, X_train, Y_train, self.params,
            scorer, train_cv, self.fit_params, self.n_trials,
            self.multiclass, self.undersampling)
        logger.info('best params: %s' % best_params)

        pipeline = self.base_pipeline
        pipeline.set_params(**best_params)
        pipeline.steps[-1] = (pipeline.steps[-1][0], self.to_second_estimator(
            pipeline.steps[-1][1], self.multiclass, self.undersampling))

        # to create model, use val_cv
        logger.info(f'get estimator with cv_select: {self.cv_select}')
        if self.cv_select == 'train_all':
            scores, pipelines = self.calc_cv_scores_estimators(
                pipeline, X_train, Y_train, scorer,
                cv=1, fit_params=self.fit_params, with_importances=True)
            score = scores[0]
            estimator = pipelines[0]
        elif self.cv_select in ['min', 'all_folds']:
            scores, pipelines = self.calc_cv_scores_estimators(
                pipeline, X_train, Y_train, scorer,
                cv=val_cv, fit_params=self.fit_params, with_importances=True)
            estimators = pipelines
            logger.info(f'cv model score mean: {np.mean(scores)}')
            logger.info(f'cv model score std: {np.std(scores)}')
            logger.info(f'cv model score max: {np.max(scores)}')
            logger.info(f'cv model score min: {np.min(scores)}')
            if self.cv_select == 'min':
                _min_index = np.array(scores).argmin()
                score = scores[_min_index]
                estimator = estimators[_min_index]
            elif self.cv_select == 'all_folds':
                _single_estimators = []
                for i, _estimator in enumerate(estimators):
                    _single_estimators.append(
                        (f'{i}_fold', _estimator))
                weights = EnsembleTrainer.get_weights(scores)

                score = np.average(scores, weights=weights)
                ensemble_trainer_obj = EnsembleTrainer(
                    X_train, Y_train, self.X_test, self.configs)
                estimator = ensemble_trainer_obj.calc_ensemble_estimator(
                    _single_estimators, ensemble_config={'mode': 'average'},
                    weights=weights, scorer=scorer)
        else:
            logger.error(f'NOT IMPLEMENTED CV SELECT: {self.cv_select}')
            raise Exception('NOT IMPLEMENTED')
        return score, estimator

    def _calc_pseudo_label_data(
        self, X_train, Y_train, estimator, classes, threshold
    ):
        _, Y_pred_proba = Outputer.predict_like(
            train_mode=self.configs['fit']['train_mode'],
            estimator=estimator, X_train=X_train, Y_train=Y_train,
            X_target=self.X_test)

        data_index, label_index = np.where(Y_pred_proba > threshold)
        pseudo_X_train = self.X_test[data_index]
        pseudo_Y_train = classes[label_index].reshape(-1, 1)
        return pseudo_X_train, pseudo_Y_train

    def _fit_with_pseudo_labeling(
        self,
        scorer, train_cv, val_cv, estimator,
        X_train, Y_train, classes, threshold
    ):
        logger.info('fit with pseudo labeling')
        pseudo_X_train, pseudo_Y_train = self._calc_pseudo_label_data(
            X_train, Y_train, estimator, classes, threshold)
        new_X_train = sp.vstack((X_train, pseudo_X_train), format='csr')
        new_Y_train = np.concatenate([Y_train, pseudo_Y_train])
        logger.info(
            'with threshold %s, train data added %s => %s'
            % (threshold, len(Y_train), len(new_Y_train)))
        return self._fit(scorer, train_cv, val_cv, new_X_train, new_Y_train)

    def _sample_with_error(self, X_train, Y_train, estimator):
        Y_pred, _ = Outputer.predict_like(
            train_mode=self.configs['fit']['train_mode'],
            estimator=estimator, X_train=X_train, Y_train=Y_train,
            X_target=X_train)

        data_index = np.where(Y_pred != self.ravel_like(Y_train))
        error_X_train = X_train[data_index]
        error_Y_train = Y_train[data_index]
        return error_X_train, error_Y_train

    def _fit_with_error_sampling(
        self, scorer, train_cv, val_cv, estimator, X_train, Y_train, score
    ):
        logger.info('fit with error_sampling')
        new_X_train, new_Y_train = self._sample_with_error(
            X_train, Y_train, estimator)
        logger.info(
            'with error_sampling, error train data is %s'
            % len(new_Y_train))
        _score, _estimator = \
            self._fit(scorer, train_cv, val_cv, new_X_train, new_Y_train)

        _single_estimators = [
            ('base', estimator),
            ('error', _estimator),
        ]
        weights = EnsembleTrainer.get_weights(
            np.array([len(Y_train), len(new_Y_train)]))

        score = np.average(np.array([score, _score]), weights=weights)
        ensemble_trainer_obj = EnsembleTrainer(
            X_train, Y_train, self.X_test, configs=self.configs)
        estimator = ensemble_trainer_obj.calc_ensemble_estimator(
            _single_estimators, ensemble_config={'mode': 'average'},
            weights=weights, scorer=scorer)
        return score, estimator

    def calc_single_estimator(
        self,
        model_config, scorer=get_scorer('accuracy'),
        train_cv=KFold(n_splits=3, shuffle=True, random_state=42),
        val_cv=KFold(n_splits=3, shuffle=True, random_state=43),
        nn_func=None, X_train=None, Y_train=None
    ):
        if X_train is None:
            X_train = self.X_train
        if Y_train is None:
            Y_train = self.Y_train
        self._get_model_params(model_config)
        self.base_pipeline = \
            self._get_base_pipeline(model_config, nn_func, X_train)
        logger.info(f'base_pipeline: {self.base_pipeline}')

        # fit
        logger.info('fit')
        score, estimator = \
            self._fit(scorer, train_cv, val_cv, X_train, Y_train)
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
                classes = np.sort(np.unique(Y_train))

            score, estimator = self._fit_with_pseudo_labeling(
                scorer, train_cv, val_cv, estimator,
                X_train, Y_train, classes, threshold)
            logger.info(f'score: {score}')
            logger.info(f'estimator: {estimator}')

        # error sampling
        if model_config.get('error_sampling'):
            if self.configs['fit']['train_mode'] == 'reg':
                logger.error('NOT IMPLEMENTED ERROR SAMPLING WITH REGRESSION')
                raise Exception('NOT IMPLEMENTED')

            score, estimator = self._fit_with_error_sampling(
                scorer, train_cv, val_cv, estimator, X_train, Y_train, score)
            logger.info(f'score: {score}')
            logger.info(f'estimator: {estimator}')

        return score, estimator
