from featurex.stimuli.text import TextStim, ComplexTextStim
from featurex.extractors import StimExtractor, ExtractorCollection
from featurex.support.exceptions import FeatureXError
from featurex.support.decorators import requires_nltk_corpus
import numpy as np
from featurex.core import Value, Event
import pandas as pd
import nltk


class TextExtractor(StimExtractor):

    ''' Base Text Extractor class; all subclasses can only be applied to text.
    '''
    target = TextStim


class ComplexTextExtractor(StimExtractor):

    ''' Base ComplexTextStim Extractor class; all subclasses can only be
    applied to ComplexTextStim instance.
    '''
    target = ComplexTextStim


class DictionaryExtractor(TextExtractor):

    ''' A generic dictionary-based extractor that supports extraction of
    arbitrary features contained in a lookup table.
    Args:
        dictionary (str): The filename of the dictionary containing the feature
            values. Format must be tab-delimited, with the first column
            containing the text key used for lookup. Subsequent columns each
            represent a single feature that can be used in extraction.
        variables (list): Optional subset of columns to keep from the
            dictionary.
        missing: Value to insert if no lookup value is found for a text token.
            Defaults to numpy's NaN.
    '''

    def __init__(self, dictionary, variables=None, missing=np.nan):
        self.data = pd.read_csv(dictionary, sep='\t', index_col=0)
        self.variables = variables
        if variables is not None:
            self.data = self.data[variables]
        # Set up response when key is missing
        self.missing = missing
        super(self.__class__, self).__init__()

    def apply(self, stim):
        if stim.text not in self.data.index:
            vals = pd.Series(self.missing, self.variables)
        else:
            vals = self.data.loc[stim.text]
        return Value(stim, self, vals.to_dict())


class LengthExtractor(TextExtractor):

    ''' Extracts the length of the text in characters. '''

    def apply(self, stim):
        return Value(stim, self, {'text_length': len(stim.text)})


class NumUniqueWordsExtractor(TextExtractor):

    ''' Extracts the number of unique words used in the text. '''

    @requires_nltk_corpus
    def apply(self, stim, tokenizer=None):
        text = stim.text
        if tokenizer is None:
            try:
                import nltk
                return len(nltk.word_tokenize(text))
            except:
                return len(text.split())
        else:
            return Value(stim, self,
                         {'num_unique_words': tokenizer.tokenize(text)})


class PartOfSpeechExtractor(ComplexTextExtractor):

    ''' Tags parts of speech in text with nltk. '''

    @requires_nltk_corpus
    def apply(self, stim):
        words = [w.text for w in stim]
        pos = nltk.pos_tag(words)
        if len(words) != len(pos):
            raise FeatureXError(
                "The number of words in the ComplexTextStim does not match "
                "the number of tagged words returned by nltk's part-of-speech"
                " tagger.")
        events = []
        for i, w in enumerate(stim):
            value = Value(stim, self, {'part_of_speech': pos[i][1]})
            event = Event(onset=w.onset, duration=w.duration, values=[value])
            events.append(event)
        return events


class BasicStatsExtractorCollection(ExtractorCollection, TextExtractor):

    ''' A collection of basic text statistics. Just a prototype; needs work.
    '''

    def __init__(self, statistics=None):

        all_stats = {'length', 'numuniquewords'}
        if statistics is not None:
            statistics = set([s.lower() for s in statistics]) & all_stats
        else:
            statistics = all_stats

        super(BasicStatsExtractorCollection, self).__init__(statistics)
