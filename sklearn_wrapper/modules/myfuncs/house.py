from keras.layers.core import Dense
from keras.models import Sequential


def create_nn_model():
    input_dim = 100
    activation = 'relu'
    output_dim = 1
    optimizer = 'adam'

    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim))

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer)
    model.summary()
    return model
