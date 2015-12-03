"""
A module for classifyng a document.
"""

from __future__ import division
import os
import codecs

import datetime
import sys

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.feature_extraction import DictVectorizer

from features import Features

from progress import progress_bar

PATH = 'Gutenberg/dataset'

def create_dataset(featext):
    """
    Reads in a set of texts and splits it into train and test sets, tagged
    by author.
    """

    files = sorted(os.listdir(PATH))

    authors = [file_name.split('___')[0] for file_name in files]
    authors = index_authors(authors)

    feature_dicts = []

    for count, file_name in enumerate(files):
        # print progress bar
        sys.stdout.write('\r%s' % progress_bar(count, len(files)))
        sys.stdout.flush()

        feature_dict = featext.extract_features(get_content(file_name))
        feature_dicts.append(feature_dict)
    print '\r%s' % progress_bar(len(files), len(files))

    # features = [feature_vector(FeatExtextract_features(get_content(file_name)))
    #             for file_name in files]

    return np.array(feature_dicts), np.array(authors)

def index_authors(authors):
    """
    Replaces a list of authors with a list of numbers corresponding to authors.
    """

    author_indices = {author: count
                      for count, author in enumerate(np.unique(authors))}
    inverse_author_indices = {count: author
                              for author, count in author_indices.items()}
    print '  author indices:'
    for i in range(len(inverse_author_indices)):
        print '    ', str(i) + ' - ' + inverse_author_indices[i]
    print
    return [author_indices[author] for author in authors]

def get_content(file_name):
    """
    Returns the contents of a file given a file name.
    """

    return codecs.open(os.path.join(PATH, file_name), 'r', 'utf8').read()

def accuracy(predictions, gold_labels):
    """
    Returns the accuracy of a list of predictions to a list of gold labels.
    """

    correct = 0.0
    for prediction, gold_label in zip(predictions, gold_labels):
        if prediction == gold_label:
            correct += 1.0

    return correct / len(gold_labels)

def main(arg):
    print

    print '---creating features extractor---'
    print '[' +  datetime.datetime.now().ctime() + ']'
    featext = Features()
    print '---created features extractor---'
    print '[' +  datetime.datetime.now().ctime() + ']'
    print

    print '---creating dataset---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    feature_dicts, authors = create_dataset(featext)
    print '---created dataset---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    print

    vec = DictVectorizer()
    vectorized_features = vec.fit_transform(feature_dicts)
    features = vectorized_features.toarray()
    print vec.get_feature_names()
    print

    np.random.seed()
    indices = np.random.permutation(len(features))

    num_test = arg

    print '---partitioning dataset---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    features_train = features[indices[:-num_test]]
    authors_train = authors[indices[:-num_test]]
    features_test = features[indices[-num_test:]]
    authors_test = authors[indices[-num_test:]]
    print '---partitioned dataset---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    print

    print '---training knn---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    knn = KNeighborsClassifier()
    knn.fit(features_train, authors_train)
    print '---trained knn---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    print

    print '---training logistic---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(features_train, authors_train)
    print '---trained logistic---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    print

    print '---training random forest---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    randomforest = RandomForestClassifier(n_estimators=100)
    randomforest.fit(features_train, authors_train)
    print '---trained random forest---',
    print '[' +  datetime.datetime.now().ctime() + ']'
    print


    # svc is computationally expensive
    # print '---training svc---',
    # print '[' +  datetime.datetime.now().ctime() + ']'
    # svc = svm.SVC(kernel='linear')
    # svc.fit(features_train, authors_train)
    # print '---trained svc---',
    # print '[' +  datetime.datetime.now().ctime() + ']'

    print

    print 'knn:          ', accuracy(knn.predict(features_test), authors_test)
    # print 'knn:          ', knn.predict(features_test)
    # print '  ', accuracy(knn.predict(features_test), authors_test)

    print 'logistic:     ', accuracy(logistic.predict(features_test), authors_test)
    # print 'logistic:     ', logistic.predict(features_test)
    # print '  ', accuracy(logistic.predict(features_test), authors_test)
    # print logistic.coef_

    print 'random forest:', accuracy(randomforest.predict(features_test), authors_test)
    # print 'random forest:', randomforest.predict(features_test)
    # print '  ', accuracy(randomforest.predict(features_test), authors_test)
    # importances = randomforest.feature_importances_

    # print 'svc:          ', svc.predict(features_test)
    # print '  ', accuracy(svc.predict(features_test), authors_test)

    # print
    # print 'gold:         ', authors_test

    return knn, logistic, randomforest, features_test, authors_test

if __name__ == '__main__':
    main(int(sys.argv[1]))
