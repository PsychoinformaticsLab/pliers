''' Filters that operate on TextStim inputs. '''

from six import string_types
from nltk import stem
from nltk.tokenize import word_tokenize
from pliers.stimuli.text import TextStim
from .base import Filter


class TextFilter(Filter):

    ''' Base class for all TextFilters. '''

    _input_type = TextStim


class WordStemmingFilter(TextFilter):

    ''' Nltk-based word stemming Filter.
    Args:
        stemmer (str, Stemmer): If a string, must be the name of one of the
            stemming modules available in nltk.stem. Valid values are
            'porter', 'snowball', 'isri', 'lancaster', 'regexp', 'wordet',
            or 'rslp'. Alternatively, an initialized nltk StemmerI instance
            can be passed.
        args, kwargs: Optional positional and keyword args passed onto the
            nltk stemmer.
    '''

    stemmers = {
        'porter': 'PorterStemmer',
        'snowball': 'SnowballStemmer',
        'lancaster': 'LancasterStemmer',
        'isri': 'ISRIStemmer',
        'regexp': 'RegexpStemmer',
        'rslp': 'RSLPStemmer'
    }

    _log_attributes = ('stemmer',)

    def __init__(self, stemmer='porter', *args, **kwargs):

        if isinstance(stemmer, string_types):
            if stemmer not in self.stemmers:
                valid = list(self.stemmers.keys())
                raise ValueError("Invalid stemmer '%s'; please use one of %s." %
                                 (stemmer, valid))
            stemmer = getattr(stem, self.stemmers[stemmer])(*args, **kwargs)
        elif not isinstance(stemmer, (stem.StemmerI, stem.WordNetLemmatizer)):
            raise ValueError("stemmer must be either a valid string, or an "
                             "instance of class StemmerI.")
        self.stemmer = stemmer
        super(WordStemmingFilter, self).__init__()

    def _filter(self, stim):
        stemmed = self.stemmer.stem(stim.text)
        return TextStim(stim.filename, stemmed, stim.onset, stim.duration)


class TokenizingFilter(TextFilter):

    ''' Tokenizes a TextStim into several word TextStims
    Args:
        tokenizer (nltk Tokenizer): a nltk Tokenizer to tokenize
            with. Will use the word_tokenize method if none is specified.
    '''

    _log_attributes = ('tokenizer',)

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        super(TokenizingFilter, self).__init__()

    def _filter(self, stim):
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(stim.text)
        else:
            tokens = word_tokenize(stim.text)
        stims = [TextStim(stim.filename, token, stim.onset, stim.duration)
                 for token in tokens]
        return stims
