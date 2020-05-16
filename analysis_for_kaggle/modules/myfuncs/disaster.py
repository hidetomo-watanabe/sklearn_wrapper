from keras.layers import GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def translate_by_tfidf(X_train, X_test, feature_columns):
    X_train = X_train.reshape(-1,)
    X_test = X_test.reshape(-1,)
    model = TfidfVectorizer(stop_words='english')
    model.fit(np.concatenate([X_train, X_test]))
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    feature_columns = [f'tfidf_{i}' for i in range(X_train.shape[1])]
    return X_train, X_test, feature_columns


def translate_by_tokenizer(X_train, X_test, feature_columns):
    vocab_size = 10000
    oov_tok = "<OOV>"
    max_length = 256
    trunc_type = 'post'
    train_sentences = X_train.reshape(-1,)
    test_sentences = X_test.reshape(-1,)

    model = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    model.fit_on_texts(train_sentences)
    train_sequences = model.texts_to_sequences(train_sentences)
    test_sequences = model.texts_to_sequences(test_sentences)
    train_padded = pad_sequences(
        train_sequences, maxlen=max_length, truncating=trunc_type)
    test_padded = pad_sequences(test_sequences, maxlen=max_length)

    X_train = train_padded
    X_test = test_padded
    return X_train, X_test, feature_columns


def create_keras_model():
    vocab_size = 10000
    embedding_dim = 128
    max_length = 256
    output_dim = 2
    optimizer = 'adam'

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model
