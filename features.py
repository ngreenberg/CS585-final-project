"""
A module for collecting features from a document.
"""

from __future__ import division
from collections import defaultdict
import string

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import cmudict

CMU_DICT = cmudict.dict() # preloaded to improve efficiency

def extract_features(doc, label):
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
    features['%s_characters per word' % label] = charcount / wordcount
    features['%s_characters per sentence' % label] = charcount / sentencecount
    features['%s_characters per paragraph' % label] = charcount / paragraphcount
    features['%s_characters per document' % label] = charcount

    # extract words features
    features['%s_words per sentence' % label] = wordcount / sentencecount
    features['%s_words per paragraph' % label] = wordcount / paragraphcount
    features['%s_words per document' % label] = wordcount

    # extract sentences features
    features['%s_sentences per paragraph' % label] = sentencecount / paragraphcount
    features['%s_sentences per document' % label] = sentencecount

    # extract paragraphs features
    features['%s_paragraphs per document' % label] = paragraphcount

    # extract syllables features
    syllablecount = 0
    for word, count in bow.iteritems():
        syllablecount += num_of_syllables(word) * count
    features['%s_syllables per word' % label] = syllablecount / wordcount
    features['%s_syllables per sentence' % label] = syllablecount / sentencecount
    features['%s_syllables per paragraph' % label] = syllablecount / paragraphcount

    return features


#########################
# Utilities

def pos_tag_doc(doc):
    """
    Assigns a part of speech tag to each word in a document and returns
    these pairings.
    """

    return pos_tag(word_tokenize(doc))

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
