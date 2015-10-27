"""
A module for collecting features from a document.
"""

from __future__ import division
from collections import defaultdict

from nltk.corpus import cmudict

def extract_features(doc):
    """
    Extract features from a document and return a dictionary of these features
    keyed by their abbreviation.
    """

    bow = tokenize_doc(doc)

    # temporary code
    for word in bow:
        print num_of_syllables(word)

####################
# Utilities

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    """

    #TODO: make work with punctuation (but ignore apostrophes)

    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = [token.lower() for token in tokens]
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

def num_of_syllables(word):
    """
    Returns the number of syllables in the input word.
    """

    return len([phoneme for phoneme in cmudict.dict()[word.lower()][0]
                if phoneme[-1].isdigit()])
