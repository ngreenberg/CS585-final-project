"""
A module for collecting features from a document.
"""

from __future__ import division
from collections import defaultdict

from nltk.corpus import cmudict

# Utilities

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    """

    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = [token.lower() for token in tokens]
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

def numsyllables(word):
    """
    Returns the number of syllables in the input word.
    """

    return len([phoneme for phoneme in cmudict.dict()[word.lower()][0]
                if phoneme[-1].isdigit()])
