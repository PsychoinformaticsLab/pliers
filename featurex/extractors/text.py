from featurex.stimuli import text
from featurex.extractors import StimExtractor, ExtractorCollection
import numpy as np
from featurex.core import Value
import pandas as pd


class TextExtractor(StimExtractor):
    target = text.TextStim


class DictionaryExtractor(TextExtractor):

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

    def apply(self, stim):
        return len(stim.text)


class NumUniqueWordsExtractor(TextExtractor):

    def apply(self, stim, tokenizer=None):
        text = stim.text
        if tokenizer is None:
            try:
                import nltk
                return len(nltk.word_tokenize(text))
            except:
                return len(text.split())
        else:
            return tokenizer.tokenize(text)


class BasicStatsExtractorCollection(ExtractorCollection, TextExtractor):

    def __init__(self, statistics=None):

        all_stats = {'length', 'numuniquewords'}
        if statistics is not None:
            statistics = set([s.lower() for s in statistics]) & all_stats
        else:
            statistics = all_stats

        super(BasicStatsExtractorCollection, self).__init__(statistics)
