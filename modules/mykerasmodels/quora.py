from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D


def create_keras_model():
    maxlen = 100
    max_features = 50000

    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, 100)(inp)
    x = CuDNNGRU(64, return_sequences=True)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam', metrics=['accuracy'])
    return model
