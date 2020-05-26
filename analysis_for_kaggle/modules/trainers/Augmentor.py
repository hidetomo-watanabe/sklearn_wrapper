from keras.preprocessing import image

import numpy as np


class Augmentor(object):
    def __init__(self, conf, batch_size=None):
        self.datagen = image.ImageDataGenerator(conf)
        self.batch_size = batch_size

    def fit(self, X, y):
        self.datagen.fit(X)
        return self

    def fit_resample(self, X, y):
        batch_size = self.batch_size if self.batch_size else len(X)
        steps = len(X) // batch_size

        self.fit(X, y)
        batch_itr = self.datagen.flow(
            X, y, batch_size=batch_size, seed=42)
        new_X = None
        new_y = None
        for _ in range(steps):
            batch = next(batch_itr)
            if isinstance(new_X, np.ndarray):
                new_X = np.append(new_X, batch[0], axis=0)
            else:
                new_X = batch[0]
            if isinstance(new_y, np.ndarray):
                new_y = np.append(new_y, batch[1], axis=0)
            else:
                new_y = batch[1]
        return new_X, new_y
