"""
A module for collecting features from a document.
"""

from __future__ import division
import string

from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict


def extract_features(doc):
    """
    Extract features from a document and returns a dictionary of these
    features keyed by their abbreviation.
    """

    features = dict()

    words = tokenize_doc_simple(doc)

    wordcount = word_count(doc)
    sentencecount = sentence_count(doc)
    paragraphcount = paragraph_count(doc)

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
    for word in words:
        numofsyllables = num_of_syllables(word)
        # if the number of syllables for a word is unknown, assume it has
        # one syllable
        syllablecount += numofsyllables if numofsyllables else 1
    features['syllables per word'] = syllablecount / wordcount
    features['syllables per sentence'] = syllablecount / sentencecount
    features['syllables per paragraph'] = syllablecount / paragraphcount

    return features


#########################
# Utilities

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

    paragraphs = doc.split("\n")
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

def num_of_syllables(word):
    """
    Returns the number of syllables in a word.
    """

    if word.lower() in cmudict.dict():
        return len([phoneme for phoneme in cmudict.dict()[word.lower()][0]
                    if phoneme[-1].isdigit()])
    else:
        return None
