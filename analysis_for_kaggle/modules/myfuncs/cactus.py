from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam


def create_nn_model():
    input_dim1 = 32
    input_dim2 = 32
    input_dim3 = 3
    output_dim = 1

    vgg16_net = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(input_dim1, input_dim2, input_dim3))
    vgg16_net.trainable = False
    model = Sequential()
    model.add(vgg16_net)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-5),
        metrics=['accuracy'])
    model.summary()
    return model
