"""
A module for collecting features from a document.
"""

import string
from collections import defaultdict

from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict


def extract_features(doc):
    """
    Extract features from a document and returns a dictionary of these
    features keyed by their abbreviation.
    """

    features = dict()

    bow = tokenize_doc_to_words(doc)

    syllables_count = 0
    for word in bow:
        syllables_count += num_of_syllables(word) * bow[word]
    features['sylpw'] = syllables_count / word_count(doc)
    features['sylps'] = syllables_count / sentence_count(doc)
    features['sylpp'] = syllables_count / paragraph_count(doc)

    return features


#########################
# Utilities

def word_count(doc):
    """
    Returns the number of words in a document as defined by
    tokenize_doc_to_words.
    """

    tokens = [word.strip(string.punctuation).lower()
              for word in doc.split(" ")]
    # remove the empty string
    return len([token for token in tokens if token])

def sentence_count(doc):
    """
    Returns the number of sentences in a document.
    """

    return len(sent_tokenize(doc))

def paragraph_count(doc):
    """
    Returns the number of paragraphs in a document.
    """

    paragraphs = doc.split("\n")
    # remove the empty string
    return len([paragraph for paragraph in paragraphs if paragraph])

def tokenize_doc_to_words(doc):
    """
    Tokenize a document and return its bag-of-words representation. Keep
    only words, removing punctuation at the beginning and end of words, and
    converting every word to lowercase.
    """

    bow = defaultdict(float)
    tokens = [word.strip(string.punctuation).lower()
              for word in doc.split(" ")]
    # remove the empty string
    tokens = [token for token in tokens if token]
    for token in tokens:
        bow[token] += 1.0
    return bow

def num_of_syllables(word):
    """
    Returns the number of syllables in a word.
    """

    if word.lower() in cmudict.dict():
        return len([phoneme for phoneme in cmudict.dict()[word.lower()][0]
                    if phoneme[-1].isdigit()])
    else:
        return None
