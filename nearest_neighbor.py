from __future__ import division
from collections import defaultdict

import os
import random
import codecs
import math
import operator

import features

path = "Gutenberg/subset"
num_feats = 0

def create_dataset(split = 0.5, size = None):
	"""
	Reads in a set of texts and split into train and test sets, tagged by author
	"""
	train_data, test_data = [],[]
	max_feats = defaultdict(float)
	for file_name in os.listdir(path)[:size]:
		base_name = os.path.basename(file_name)
		author = base_name.split('_',1)[0]

		print "Reading in from %s" % base_name
		with codecs.open(os.path.join(path, file_name),'r','utf8') as doc:
			content = doc.read()
			feat_vec = features.extract_features(content)
			for feat, value in feat_vec.iteritems():
				if value > max_feats[feat]:
					max_feats[feat]=value
			length = len(feat_vec)
			feat_vec['author']=author
			if random.random()<split:
				train_data.append(feat_vec)
			else:
				test_data.append(feat_vec)

	print "Normalizing..."
	for feat_vec in train_data:
		for feat in feat_vec:
			feat_vec[feat]/=max_feats[feat]
	for feat_vec in test_data:
		for feat in feat_vec:
			feat_vec[feat]/=max_feats[feat]

	return train_data, test_data


def distance(vec1, vec2):
	"""
	Finds the simple distance- the root of the sum of the squares of the distances for each feature
	"""
	sum = 0
	for x in range(num_feats):
		sum += pow(vec1[x]-vec2[x], 2)
	return math.sqrt(sum)

def get_K_neighbors(train_data, test_instance, k):
	"""
	Finds the K instances in the train data that are nearest to the test instance
	"""
	distances = []
	neighbors = []
	for train_instance in train_data:
		distance = (distance(train_instance, test_instance))
		author = train_instance['author']
		distances.append[(distance, author)]
	distances.sort(key=operator.itemgetter(1))
	for x in range(k):
		neighbors.append(distances[x][1])
	return neighbors

def classify_instance(train_data, test_instance, k):
	author_votes = defaultdict(float)
	neighbors = get_K_neighbors(train_data, test_instance, k)
	for author in neighbors:
		author_votes[author]+=1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def test(train_data, test_data):
	print "Testing..."
	k = math.sqrt(len(train_data))
	correct, total = 0,0
	for instance in test_data:
		pred_author = classify_instance(train_data, instance, k)
		if pred_author == instance[-1]:
			correct+=1
		total+=1
	print "Accuracy: %d/%d = %.3f" % (correct, total, (correct/total))

if __name__=='__main__':
	training_set, testing_set = create_dataset(0.5, 100)
	test(training_set, testing_set)
