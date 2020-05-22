from keras.preprocessing import image


class Augmentor(object):
    def __init__(self, conf):
        self.datagen = image.ImageDataGenerator(conf)

    def fit(self, X, y):
        self.datagen.fit(X)
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        batch_itr = self.datagen.flow(X, y, batch_size=len(X), seed=42)
        batch = next(batch_itr)
        return batch[0], batch[1]
