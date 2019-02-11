from keras.models import Sequential
from keras.layers.core import Dense


def create_keras_model():
    # input_dim = self.X_train.shape[1]
    input_dim = 100
    activation = 'relu'
    # output_dim = len(np.unique(self.Y_train))
    output_dim = 1
    optimizer = 'adam'

    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim))

    # compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
