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
