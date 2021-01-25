import copy
import types
from logging import getLogger

from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import numpy as np

from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Sequential


logger = getLogger('predict').getChild('MyKeras')


class MyKerasClassifier(KerasClassifier):
    def fit(
        self, x, y,
        with_generator=False, generator=None,
        batch_size=None, validation_data=None, **kwargs
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
            fit_args['steps_per_epoch'] = len(x) // batch_size
            fit_args['validation_steps'] = \
                len(validation_data[0]) // batch_size
            logger.info(f'with generator: {fit_args}')
            generator.fit(x)
            history = self.model.fit_generator(
                generator.flow(x, y, batch_size=batch_size, seed=42),
                validation_data=generator.flow(
                    validation_data[0], validation_data[1],
                    batch_size=batch_size, seed=42),
                **fit_args)
        else:
            history = self.model.fit(x, y, **fit_args)
        return history


class MyKerasRegressor(KerasRegressor):
    def fit(
        self, x, y,
        with_generator=False, generator=None,
        batch_size=None, validation_data=None, **kwargs
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
            fit_args['steps_per_epoch'] = len(x) // batch_size
            fit_args['validation_steps'] = \
                len(validation_data[0]) // batch_size
            logger.info(f'with generator: {fit_args}')
            generator.fit(x)
            history = self.model.fit_generator(
                generator.flow(x, y, batch_size=batch_size, seed=42),
                validation_data=generator.flow(
                    validation_data[0], validation_data[1],
                    batch_size=batch_size, seed=42),
                **fit_args)
        else:
            history = self.model.fit(x, y, **fit_args)
        return history
