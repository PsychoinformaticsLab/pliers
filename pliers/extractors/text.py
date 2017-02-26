'''
Extractors that operate primarily or exclusively on Text stimuli.
'''

from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.support.exceptions import PliersError
from pliers.support.decorators import requires_nltk_corpus
from pliers.datasets.text import fetch_dictionary
import numpy as np
import pandas as pd
from six import string_types

# Optional dependencies
try:
    import nltk
except ImportError:
    nltk is None


class TextExtractor(Extractor):

    ''' Base Text Extractor class; all subclasses can only be applied to text.
    '''
    _input_type = TextStim


class ComplexTextExtractor(Extractor):

    ''' Base ComplexTextStim Extractor class; all subclasses can only be
    applied to ComplexTextStim instance.
    '''
    _input_type = ComplexTextStim

    def _extract(self, stim):
        ''' Returns all words. '''
        props = [(e.text, e.onset, e.duration) for e in stim.elements]
        vals, onsets, durations = map(list, zip(*props))
        return ExtractorResult(vals, stim, self, ['word'], onsets, durations)


class DictionaryExtractor(TextExtractor):

    ''' A generic dictionary-based extractor that supports extraction of
    arbitrary features contained in a lookup table.
    Args:
        dictionary (str, DataFrame): The dictionary containing the feature
            values. Either a string giving the path to the dictionary file,
            or a pandas DF. Format must be tab-delimited, with the first column
            containing the text key used for lookup. Subsequent columns each
            represent a single feature that can be used in extraction.
        variables (list): Optional subset of columns to keep from the
            dictionary.
        missing: Value to insert if no lookup value is found for a text token.
            Defaults to numpy's NaN.
    '''

    _log_attributes = ('dictionary', 'variables', 'missing')

    def __init__(self, dictionary, variables=None, missing=np.nan):
        if isinstance(dictionary, string_types):
            self.dictionary = dictionary  # for TranformationHistory logging
            dictionary = pd.read_csv(dictionary, sep='\t', index_col=0)
        else:
            self.dictionary = None
        self.data = dictionary
        if variables is None:
            variables = list(self.data.columns)
        else:
            self.data = self.data[variables]
        self.variables = variables
        # Set up response when key is missing
        self.missing = missing
        super(DictionaryExtractor, self).__init__()

    def _extract(self, stim):
        if stim.text not in self.data.index:
            vals = pd.Series(self.missing, self.variables)
        else:
            vals = self.data.loc[stim.text].fillna(self.missing)
        vals = vals.to_dict()
        return ExtractorResult(np.array([list(vals.values())]), stim, self,
                               features=list(vals.keys()))


class PredefinedDictionaryExtractor(DictionaryExtractor):

    ''' A generic Extractor that maps words onto values via one or more
    pre-defined dictionaries accessed via the web.
    Args:
        variables (list or dict): A specification of the dictionaries and
            column names to map the input TextStims onto. If a list, each
            element must be a string with the format 'dict/column', where the
            value before the slash gives the name of the dictionary, and the
            value after the slash gives the name of the column in that
            dictionary. These names can be found in the dictionaries.json
            specification file under the datasets submodule. Examples of
            valid values are 'affect/V.Mean.Sum' and
            'subtlexusfrequency/Lg10WF'. If a dict, the keys are the names of
            the dictionary files (e.g., 'affect'), and the values are lists
            of columns to use (e.g., ['V.Mean.Sum', 'V.SD.Sum']).
        missing (object): Value to use when an entry for a word is missing in
            a dictionary (defaults to numpy's NaN).
        case_sensitive (bool): If True, entries in the dictionary are treated
            as case-sensitive (e.g., 'John' and 'john' are different words).
    '''

    _log_attributes = ('variables', 'missing', 'case_sensitive')

    def __init__(self, variables, missing=np.nan, case_sensitive=True):

        self.case_sensitive = case_sensitive

        if isinstance(variables, (list, tuple)):
            _vars = {}
            for v in variables:
                v = v.split('/')
                if v[0] not in _vars:
                    _vars[v[0]] = []
                if len(v) == 2:
                    _vars[v[0]].append(v[1])
            variables = _vars

        dicts = []
        for k, v in variables.items():
            d = fetch_dictionary(k)
            if not case_sensitive:
                d.index = d.index.str.lower()
            if v:
                d = d[v]
            d.columns = ['%s_%s' % (k, c) for c in d.columns]
            dicts.append(d)

        dictionary = pd.concat(dicts, axis=1, join='outer')
        super(PredefinedDictionaryExtractor, self).__init__(
            dictionary, missing=missing)


class LengthExtractor(TextExtractor):

    ''' Extracts the length of the text in characters. '''

    def _extract(self, stim):
        return ExtractorResult(np.array([[len(stim.text.strip())]]), stim,
                               self, features=['text_length'])


class NumUniqueWordsExtractor(TextExtractor):

    ''' Extracts the number of unique words used in the text. '''

    _log_attributes = ('tokenizer',)

    def __init__(self, tokenizer=None):
        TextExtractor.__init__(self)
        self.tokenizer = tokenizer

    @requires_nltk_corpus
    def _extract(self, stim):
        text = stim.text
        if self.tokenizer is None:
            if nltk is None:
                num_words = len(set(text.split()))
            else:
                num_words = len(set(nltk.word_tokenize(text)))
        else:
            num_words = len(set(self.tokenizer.tokenize(text)))

        return ExtractorResult(np.array([[num_words]]), stim, self,
                               features=['num_unique_words'])


class PartOfSpeechExtractor(ComplexTextExtractor):

    ''' Tags parts of speech in text with nltk. '''

    @requires_nltk_corpus
    def _extract(self, stim):
        words = [w.text for w in stim]
        pos = nltk.pos_tag(words)
        if len(words) != len(pos):
            raise PliersError(
                "The number of words in the ComplexTextStim does not match "
                "the number of tagged words returned by nltk's part-of-speech"
                " tagger.")

        data = {}
        onsets = []
        durations = []
        for i, w in enumerate(stim):
            p = pos[i][1]
            if p not in data:
                data[p] = [0] * len(words)
            data[p][i] += 1
            onsets.append(w.onset)
            durations.append(w.duration)

        return ExtractorResult(np.array(list(data.values())).transpose(),
                               stim, self, features=list(data.keys()),
                               onsets=onsets, durations=durations)
