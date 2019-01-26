from keras.models import Sequential
from keras.layers.core import Dense


def create_keras_model():
    # input_dim = self.X_train.shape[1]
    input_dim = 9
    activation = 'relu'
    middle_dim = 100
    # output_dim = len(np.unique(self.Y_train))
    output_dim = 2
    optimizer = 'adam'

    model = Sequential()
    # first layer
    model.add(Dense(middle_dim, input_dim=input_dim, activation=activation))
    model.add(Dense(middle_dim, activation=activation))
    # last layer
    model.add(Dense(output_dim, activation="softmax"))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer, metrics=['accuracy'])
    return model
