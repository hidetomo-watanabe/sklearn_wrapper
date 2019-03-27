from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D


def create_keras_model():
    # input_dim = self.X_train.shape[1]
    input_dim = 100
    n_pool1 = 4
    n_pool2 = 5
    n_pool3 = 5
    n_kernel = 5
    n_filter = 10
    # output_dim = self.Y_train.shape[1]
    output_dim = 1

    model = Sequential()
    model.add(Conv1D(
        input_dim, n_kernel, padding='same',
        input_shape=(input_dim, 1), activation='relu'))
    model.add(MaxPooling1D(n_pool1, padding='same'))
    model.add(Conv1D(input_dim, n_kernel, padding='same', activation='relu'))
    model.add(MaxPooling1D(n_pool2, padding='same'))
    model.add(Conv1D(n_filter, n_kernel, padding='same', activation='relu'))
    model.add(MaxPooling1D(n_pool3, padding='same'))
    model.add(Conv1D(output_dim, n_kernel, padding='same', activation='tanh'))

    # compile model
    model.compile(
        loss="mean_squared_error",
        optimizer='adam')
    model.summary()
    return model
