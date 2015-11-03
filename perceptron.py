from __future__ import division
from collections import defaultdict

import os
import random

import features

authors = []
path = "Gutenberg/txt"

def create_dataset(size=None):
	"""
	Reads in a set of texts and split into train and test sets, tagged by author
	"""
        if size is None:
                size = len(os.listdir(path))
	train, test = [],[]
	for file_name in os.listdir(path)[:size]:
		author = os.path.basename(file_name).split('-',1)[0]
		with open(os.path.join(path, file_name),'r') as doc:
			content = doc.read().decode('utf-8')
			feat_vec = features.extract_features(content)
			authors.append(author)
			if random.randint(0,1):
				train.append((feat_vec, author))
			else:
				test.append((feat_vec, author))
	return train, test


def train(train, testdata, stepsize=1, numpasses=10):
	"""
	Trains the classifier over a series of passes, and tests at each iteration
	"""
	weights = defaultdict(float)

	for iteration in range(numpasses):
		print "Training iteration %d" % iteration
		random.shuffle(train)
		for feat_vec, author in train:
			pred_author = predict_author(feat_vec,weights)
			if pred_author != author:
				labeled_feat_vec = label_feat_vector(feat_vec, author)
				for feat, weight in labeled_feat_vec.iteritems():
					weights[feat]+=stepsize*weight

		print "Testing iteration %d" % iteration
		test(testdata, weights)
	return weights

def test(docs, weights):
	"""
	Tests classifier on test set
	"""
	correct, total = 0,0
	for doc, author in docs:
		pred_author = predict_author(doc, weights)
		if pred_author == author:
			correct += 1
		total += 1
	print "		%d/%d = %.4f accuracy" % (correct, total, correct/total)


def predict_author(feat_vec, weights):
	"""
	Finds the author with the highest score under the model
	"""
	scores = defaultdict(float)
	for author in authors:
		author_feat_vec = label_feat_vector(feat_vec, author)
		scores[author] = dict_dotprod(author_feat_vec , weights)
	return dict_argmax(scores)

def label_feat_vector(feat_vec, label):
	labeled_feat_vec = dict()
	for tag, weight in feat_vec.iteritems():
		labeled_feat_vec['%s_%s' % (label, tag)] = weight
	return labeled_feat_vec


###################################
# Utilities

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<d2 else d2
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

if __name__=='__main__':
	training_set, testing_set = create_dataset(size=100)
	train(training_set, testing_set)
