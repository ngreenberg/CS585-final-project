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
        tokens = self.pos_tag_doc(doc)

        pos_counts = self.vectorize_pos_tags(tokens)
        poswordcount = sum(pos_counts.values())
        for i in xrange(82, 101):
            features['%d per word' % i] = pos_counts[i] / poswordcount


        # extract noun phrase features
        phrase_sum = 0
        phrase_count = 0
        noun_chunks = tokens.noun_chunks
        for chunk in noun_chunks:
            phrase = chunk.orth_
            if not phrase.isspace():
                phrase_sum += len(chunk)
                phrase_count += 1
        features['words per noun phrase'] = phrase_sum / phrase_count
        
        # extract character trigram frequency features
        ngramfreqs = defaultdict(float)
        for word, count in bow.iteritems():
            trigrams = zip(word, word[1:], word[2:])
            for instance in trigrams:
                ngramfreqs[instance]+=count
        sortedfreqs = sorted(ngramfreqs.items, key = operator.itemgetter(1), reverse = True)
        add_stat_features(sortedfreqs[:300], 'char trigram frequency')

        #extract word and POS trigram freqency features
        POSfreqs = defaultdict(float)
        wordfreqs = defaultdict(float)
        sentences = sent_tokenize(doc)
        for sent in sentences:
            tags = {}
            words = {}
            sentPOS = self.pos_tag_doc(sent)
            for word, tag in sentPOS.itervalues():
                tags.append(tag)
                words.append(word)
            POStrigrams = zip(tags, tags[1:], tags[2:])
            wordtrigrams = zip(words, words[1:], words[2:])
            for instance in POStrigrams:
                POSfreqs[instance]+=1
            for instance in wordtrigrams:
                wordfreqs[instance]+=1
        sortedPOS = sorted(POSfreqs.values, reverse = True)
        sortedwords = sorted(wordfreqs.values, reverse = True)
        add_stat_features(sortedwords[:300], 'word trigram frequency')
        add_stat_features(sortedPOS[:100], 'POS trigram frequency')
        return features

        def add_stat_features(values, name):
            features['max %s' % name] = values[0]
            features['min %s' % name] = values[-1]
            features['%s mean' % name] = mean(values)
            features['%s variance' % name] = variance(values)
        
        return features

    #########################
    # Utilities
    
    def mean(values):
        """
        Returns the mean of a list of values
        """

        mean = sum(values)/len(values)
        return mean

    def variance(values):
        """
        Returns the variance of a list of values
        """

        variance = 0
        mean = mean(values)
        for x in range(length):
            variance += (values[x]-mean)**2
        variance/=len(values)
        return variance

    def pos_tag_doc(self, doc):
        """
        Assigns a part of speech tag to each word in a document and returns
        these pairings.
        """

        return self.spacy(doc)

    def vectorize_pos_tags(self, tokens):
        """
        Creates a count of part of speech tags for a document.
        """

        pos_counts = defaultdict(float)
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
