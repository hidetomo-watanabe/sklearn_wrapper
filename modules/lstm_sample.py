from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam


def create_keras_model():
    # input_dim = self.X_train.shape[1]
    input_dim = 100
    n_hidden = 500
    # output_dim = self.Y_train.shape[1]
    output_dim = 1

    model = Sequential()
    model.add(LSTM(
        n_hidden,
        batch_input_shape=(None, input_dim, output_dim),
        return_sequences=False))
    model.add(Dense(output_dim))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.001)

    # compile model
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer)
    return model
