from __future__ import division
from collections import defaultdict

import os
import random

import features

authors = []
path = "C:/Users/Patrick/Downloads/Gutenberg/txt"

def create_dataset():
	"""
	Reads in a set of texts and split into train and test sets, tagged by author
	"""
	train, test = [],[]
	for file_name in os.listdir(path):
		with open(os.path.join(path, file_name),'r') as doc:
			author = os.path.basename(doc.name).split('-',1)[0]
			content = doc.read()
			authors.append(author)
			if random.randint(0,1):
				train.append((content, author))
			else:
				test.append((content, author))
	return train, test


def train(train, test, stepsize=1, numpasses=10):
	"""
	Trains the classifier over a series of passes, and tests at each iteration
	"""
	weights = defaultdict(float)

	for iteration in range(numpasses):
		print "Training iteration %d" % iteration
		random.shuffle(train)
		for doc, author in train:
			pred_author = predict_author(doc,weights)
			if pred_author != author:
				feat_vec = features.extract_features(doc, author)
				for feature, weight in feat_vec.iteritems():
					weights[feat]+=stepsize*weight

		print "Testing iteration %d" % iteration
		test(test, weights)
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


def predict_author(doc, weights):
	"""
	Finds the author with the highest score under the model
	"""
	scores = defaultdict(float)
	for author in authors:
		author_feat_vec = features.extract_features(doc, author)
		scores[label] = dict_dotprod(author_feat_vec,weights)
	return dict_argmax(scores)


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
	training_set, testing_set = create_dataset()
	train(training_set, testing_set)
