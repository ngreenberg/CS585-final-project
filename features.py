"""
A module for collecting features from a document.
"""

from __future__ import division
from collections import defaultdict
import string

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import cmudict

import spacy.en

CMU_DICT = cmudict.dict() # preloaded to improve efficiency

def extract_features(doc, nlp):
    """
    Extract features from a document and returns a dictionary of these
    features keyed by their abbreviation and document label.
    """

    features = dict()

    bow = vectorize_doc_simple(doc)

    charcount = char_count(doc)
    wordcount = word_count(doc)
    sentencecount = sentence_count(doc)
    paragraphcount = paragraph_count(doc)

    # extract characters features
    features['characters per word'] = charcount / wordcount
    features['characters per sentence'] = charcount / sentencecount
    features['characters per paragraph'] = charcount / paragraphcount
    features['characters per document'] = charcount

    # extract words features
    features['words per sentence'] = wordcount / sentencecount
    features['words per paragraph'] = wordcount / paragraphcount
    features['words per document'] = wordcount

    # extract sentences features
    features['sentences per paragraph'] = sentencecount / paragraphcount
    features['sentences per document'] = sentencecount

    # extract paragraphs features
    features['paragraphs per document'] = paragraphcount

    # extract syllables features
    syllablecount = 0
    for word, count in bow.iteritems():
        syllablecount += num_of_syllables(word) * count
    features['syllables per word'] = syllablecount / wordcount
    features['syllables per sentence'] = syllablecount / sentencecount
    features['syllables per paragraph'] = syllablecount / paragraphcount

    # extract part of speech features
    tokens = nlp(doc, tag=True, parse=False)
    pos_counts = defaultdict(float)
    for token in tokens:
        pos_counts[token.pos] += 1.0
    # pos_counts = vectorize_pos_tags(doc)
    poswordcount = sum(pos_counts.values())
    for i in xrange(82, 101):
        features['%d per word' % i] = pos_counts[i] / poswordcount
    # features['97 per word'] = pos_counts[97] / poswordcount
    # features['82 per word'] = pos_counts[82] / poswordcount
    # features['83 per word'] = pos_counts[83] / poswordcount
    # features['87 per word'] = pos_counts[87] / poswordcount
    # features['89 per word'] = pos_counts[89] / poswordcount
    # features['94 per word'] = pos_counts[94] / poswordcount
    # features['NN per word'] = pos_counts['NN'] / poswordcount
    # features['NNS per word'] = pos_counts['NNS'] / poswordcount
    # features['IN per word'] = pos_counts['IN'] / poswordcount
    # features['DT per word'] = pos_counts['DT'] / poswordcount
    # features['JJ per word'] = pos_counts['JJ'] / poswordcount
    # features['RB per word'] = pos_counts['RB'] / poswordcount
    # features['PRP per word'] = pos_counts['PRP'] / poswordcount
    # features['CC per word'] = pos_counts['CC'] / poswordcount

    return features


#########################
# Utilities

def pos_tag_doc(doc):
    """
    Assigns a part of speech tag to each word in a document and returns
    these pairings.
    """

    return pos_tag(word_tokenize(doc))

def vectorize_pos_tags(doc):
    """
    Creates a count of part of speech tags for a document.
    """

    pos_counts = defaultdict(float)
    words = pos_tag_doc(doc)
    for word, tag in words:
        pos_counts[tag] += 1.0
    return pos_counts

def char_count(doc):
    """
    Returns the number of characters in a document.
    """

    return len(doc)

def word_count(doc):
    """
    Returns the number of words in a document as defined by
    tokenize_doc_simple.
    """

    return len(tokenize_doc_simple(doc))

def sentence_count(doc):
    """
    Returns the number of sentences in a document.
    """

    return len(sent_tokenize(doc))

def paragraph_count(doc):
    """
    Returns the number of paragraphs in a document.
    """

    paragraphs = doc.split("\n\n")
    # remove the empty string
    return len([paragraph for paragraph in paragraphs if paragraph])

def tokenize_doc_simple(doc):
    """
    Tokenize a document. Keep only words, removing punctuation at the
    beginning and end of words.
    """

    tokens = [word.strip(string.punctuation) for word in doc.split()]
    # remove the empty string
    return [token for token in tokens if token]

def vectorize_doc_simple(doc):
    """
    Returns the word vector of a document. Keep only words, removing
    punctuation at the beginning and end of words, and converting all words
    to lowercase.
    """

    bow = defaultdict(float)
    tokens = [token.lower() for token in doc.split()]
    for token in tokens:
        bow[token] += 1.0
    return bow

def num_of_syllables(word):
    """
    Returns the number of syllables in a word.
    """

    if word.lower() in CMU_DICT:
        return len([phoneme for phoneme in CMU_DICT[word.lower()][0]
                    if phoneme[-1].isdigit()])
    # If word is unknown, assume 1 syllable/3 letters (average for English)
    else:
        return len(word)//3
