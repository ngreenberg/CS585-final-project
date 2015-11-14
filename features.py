"""
A module for collecting features from a document.
"""

from __future__ import division
from collections import defaultdict
import string

from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict

import spacy.en

class Features(object):
    """
    A class for collecting features from a document. Instantiation is
    necessary to improve efficiency.
    """

    def __init__(self):
        # preloaded to improve efficiency
        self.cmu_dict = cmudict.dict()
        self.spacy = spacy.en.English()

    def extract_features(self, doc):
        """
        Extract features from a document and returns a dictionary of these
        features keyed by their abbreviation and document label.
        """

        features = dict()

        bow = self.vectorize_doc_simple(doc)

        charcount = self.char_count(doc)
        wordcount = self.word_count(doc)
        sentencecount = self.sentence_count(doc)
        paragraphcount = self.paragraph_count(doc)

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
            syllablecount += self.num_of_syllables(word) * count
        features['syllables per word'] = syllablecount / wordcount
        features['syllables per sentence'] = syllablecount / sentencecount
        features['syllables per paragraph'] = syllablecount / paragraphcount

        # extract part of speech features
        pos_counts = self.vectorize_pos_tags(doc)
        poswordcount = sum(pos_counts.values())
        for i in xrange(82, 101):
            features['%d per word' % i] = pos_counts[i] / poswordcount

        return features


    #########################
    # Utilities

    def pos_tag_doc(self, doc):
        """
        Assigns a part of speech tag to each word in a document and returns
        these pairings.
        """

        return self.spacy(doc, tag=True, parse=False)

    def vectorize_pos_tags(self, doc):
        """
        Creates a count of part of speech tags for a document.
        """

        pos_counts = defaultdict(float)
        tokens = self.pos_tag_doc(doc)
        for token in tokens:
            pos_counts[token.pos] += 1.0
        return pos_counts

    def char_count(self, doc):
        """
        Returns the number of characters in a document.
        """

        return len(doc)

    def word_count(self, doc):
        """
        Returns the number of words in a document as defined by
        tokenize_doc_simple.
        """

        return len(self.tokenize_doc_simple(doc))

    def sentence_count(self, doc):
        """
        Returns the number of sentences in a document.
        """

        return len(sent_tokenize(doc))

    def paragraph_count(self, doc):
        """
        Returns the number of paragraphs in a document.
        """

        paragraphs = doc.split("\n\n")
        # remove the empty string
        return len([paragraph for paragraph in paragraphs if paragraph])

    def tokenize_doc_simple(self, doc):
        """
        Tokenize a document. Keep only words, removing punctuation at the
        beginning and end of words.
        """

        tokens = [word.strip(string.punctuation) for word in doc.split()]
        # remove the empty string
        return [token for token in tokens if token]

    def vectorize_doc_simple(self, doc):
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

    def num_of_syllables(self, word):
        """
        Returns the number of syllables in a word.
        """

        if word.lower() in self.cmu_dict:
            return len([phoneme for phoneme in self.cmu_dict[word.lower()][0]
                        if phoneme[-1].isdigit()])
        # If word is unknown, assume 1 syllable/3 letters (average for English)
        else:
            return len(word)//3
