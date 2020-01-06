import os
import sys
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam


def create_nn_model():
    input_dim1 = 32
    input_dim2 = 32
    input_dim3 = 3
    output_dim = 2

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


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]

    # train
    train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
    img_paths = []
    for img_path in tqdm(train_df['id'].to_numpy()):
        img_path = f'{DATA_PATH}/train/{img_path}'
        img_paths.append(img_path)
    train_df['img_path'] = img_paths
    train_df.to_csv(f'{DATA_PATH}/train2.csv', index=False)

    # test
    p = Path(DATA_PATH)
    ids = []
    img_paths = []
    for img_path in tqdm(p.glob(f'test/*')):
        ids.append(os.path.basename(img_path))
        img_paths.append(img_path)
    test_df = pd.DataFrame(ids, columns=['id'])
    test_df['img_path'] = img_paths
    test_df.to_csv(f'{DATA_PATH}/test2.csv', index=False)
