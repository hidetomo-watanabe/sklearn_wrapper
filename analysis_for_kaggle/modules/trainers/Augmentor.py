from keras.preprocessing import image

import numpy as np


class Augmentor(object):
    def __init__(self, conf):
        self.datagen = image.ImageDataGenerator(conf)

    def fit(self, X, y):
        self.datagen.fit(X)
        return self

    def fit_resample(self, X, y, ratio=1):
        self.fit(X, y)
        batch_itr = self.datagen.flow(
            X, y, batch_size=len(X), seed=42)
        new_X = None
        new_y = None
        for _ in range(ratio):
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
