import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam


def add_image_id(train_df, test_df):
    for df in [train_df, test_df]:
        df['ImageId'] = np.arange(1, len(df) + 1)
    return train_df, test_df


def translate_label2upper(Y_pred, Y_pred_proba):
    Y_pred = Y_pred.rename(columns={'label': 'Label'})
    return Y_pred, Y_pred_proba


def create_keras_model():
    input_dim = 100
    n_hidden = 500
    output_dim = 10

    model = Sequential()
    model.add(LSTM(
        n_hidden,
        batch_input_shape=(None, input_dim, output_dim),
        return_sequences=False))
    model.add(Dense(output_dim))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.001)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer)
    model.summary()
    return model
