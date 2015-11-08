"""
A module for classifyng a document.
"""

from __future__ import division
import os
import codecs

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm

from features import extract_features

PATH = 'Gutenberg/dataset'

def create_dataset():
    """
    Reads in a set of texts and splits it into train and test sets, tagged
    by author.
    """

    files = sorted(os.listdir(PATH))

    authors = [file_name.split('___')[0] for file_name in files]
    authors = index_authors(authors)
    features = [feature_vector(extract_features(get_content(file_name)))
                for file_name in files]

    return np.array(features), np.array(authors)

def index_authors(authors):
    """
    Replaces a list of authors with a list of numbers corresponding to authors.
    """

    author_indices = {author: count
                      for count, author in enumerate(np.unique(authors))}
    return [author_indices[author] for author in authors]

def get_content(file_name):
    """
    Returns the contents of a file given a file name.
    """

    return codecs.open(os.path.join(PATH, file_name), 'r', 'utf8').read()

def feature_vector(features):
    """
    Returns a feature vector given a dictionary of features.
    """

    return features.values()

def accuracy(predictions, gold_labels):
    """
    Returns the accuracy of a list of predictions to a list of gold labels.
    """

    correct = 0.0
    for prediction, gold_label in zip(predictions, gold_labels):
        if prediction == gold_label:
            correct += 1.0

    return correct / len(gold_labels)

if __name__ == '__main__':
    print '---creating dataset---'
    features, authors = create_dataset()
    print '---created dataset---'

    np.random.seed()
    indices = np.random.permutation(len(features))

    print '---partitioning dataset---'
    features_train = features[indices[:-20]]
    authors_train = authors[indices[:-20]]
    features_test = features[indices[-20:]]
    authors_test = authors[indices[-20:]]
    print '---partitioned dataset---'

    print '---training knn---'
    knn = KNeighborsClassifier()
    knn.fit(features_train, authors_train)
    print '---trained knn---'

    print '---training logistic---'
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(features_train, authors_train)
    print '---trained logistic---'

    print '---training svc---'
    svc = svm.SVC(kernel='linear')
    svc.fit(features_train, authors_train)
    print '---trained svc---'

    print 'knn:     ', knn.predict(features_test)
    print '  ', accuracy(knn.predict(features_test), authors_test)

    print 'logistic:', logistic.predict(features_test)
    print '  ', accuracy(logistic.predict(features_test), authors_test)

    print 'svc:     ', svc.predict(features_test)
    print '  ', accuracy(svc.predict(features_test), authors_test)

    print
    print 'gold:    ', authors_test
