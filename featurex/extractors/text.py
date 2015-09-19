from featurex import stimuli
from featurex.extractors import StimExtractor
import numpy as np
from featurex.core import Value
import pandas as pd


class TextExtractor(StimExtractor):
    target = stimuli.text.TextStim


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
