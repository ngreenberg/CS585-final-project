from __future__ import division
from collections import defaultdict

import os
import random
import codecs

import features

authors = []
path = "Gutenberg/txt"

def create_dataset(size=None):
	"""
	Reads in a set of texts and split into train and test sets, tagged by author
	"""
	do_train = 1
	train, test = [],[]
	for file_name in os.listdir(path)[:size]:
		base_name = os.path.basename(file_name)
		author = base_name.split('_',1)[0]
		if not author in authors:
			authors.append(author)

		print "Reading in from %s" % base_name
		with codecs.open(os.path.join(path, file_name),'r','utf8') as doc:
			content = doc.read()
			feat_vec = features.extract_features(content)
			if do_train:
				train.append((feat_vec, author))
				do_train = 0
			else:
				test.append((feat_vec, author))
				do_train = 1
	return train, test


def train_classifier(train, test, stepsize=1, numpasses=10):
	"""
	Trains the classifier over a series of passes, and tests at each iteration
	"""
	weights = defaultdict(float)
	weightSums = defaultdict(float)
	avg_weights = defaultdict(float)

	doc_count = 0
	
	for iteration in range(numpasses):
		print "Training iteration %d" % iteration
		random.shuffle(train)
		for feat_vec, author in train:
			doc_count+=1
			pred_author = predict_author(feat_vec, weights)
			if pred_author != author:
				labeled_feat_vec = label_feat_vector(feat_vec, author)
				pred_feat_vec = label_feat_vector(feat_vec, pred_author)
				feat_delta = dict_subtract(labeled_feat_vec, pred_feat_vec)
				for feat, value in feat_delta.iteritems():
					weights[feat]+=stepsize*value
					weightSums[feat] += (doc_count-1) * stepsize * value

		for feat in weightSums:
			avg_weights[feat] = weights[feat] - 1/doc_count*weightSums[feat] 
		print "Testing iteration %d" % iteration
		
		test_classifier(test, avg_weights)

	return avg_weights

def test_classifier(test, weights):
	"""
	Tests classifier on test set
	"""
	correct, total = 0,0
	for feat_vec, author in test:
		pred_author = predict_author(feat_vec, weights)
		if pred_author == author:
			correct += 1
		total += 1
	print "		%d/%d = %.3f accuracy" % (correct, total, correct/total)


def predict_author(feat_vec, weights):
	"""
	Finds the author with the highest score under the model
	"""
	scores = dict()
	for author in authors:
		author_feat_vec = label_feat_vector(feat_vec, author)
		scores[author] = dict_dotprod(author_feat_vec, weights)
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

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

if __name__=='__main__':
	training_set, testing_set = create_dataset(100)
	train_classifier(training_set, testing_set)
