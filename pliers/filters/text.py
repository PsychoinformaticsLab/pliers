''' Filters that operate on TextStim inputs. '''

import nltk
import string
import re

from six import string_types
from nltk import stem
from nltk.tokenize import word_tokenize
from nltk.tokenize import * # noqa
from nltk.tokenize.api import TokenizerI
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
        tokenize (bool): if True, apply the stemmer to each token in the
            TextStim, otherwise treat the whole TextStim as one token to stem.
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

    _log_attributes = ('stemmer', 'tokenize')

    def __init__(self, stemmer='porter', tokenize=True, *args, **kwargs):

        if isinstance(stemmer, string_types):
            if stemmer not in self.stemmers:
                valid = list(self.stemmers.keys())
                raise ValueError("Invalid stemmer '%s'; please use one of %s."
                                 % (stemmer, valid))
            stemmer = getattr(stem, self.stemmers[stemmer])(*args, **kwargs)
        elif not isinstance(stemmer, (stem.StemmerI, stem.WordNetLemmatizer)):
            raise ValueError("stemmer must be either a valid string, or an "
                             "instance of class StemmerI.")
        self.stemmer = stemmer
        self.tokenize = tokenize
        super(WordStemmingFilter, self).__init__()

    def _filter(self, stim):
        if self.tokenize:
            tokens = stim.text.split()
            stemmed = ' '.join([self.stemmer.stem(tok) for tok in tokens])
        else:
            stemmed = self.stemmer.stem(stim.text)
        return TextStim(stim.filename, stemmed)


class TokenizingFilter(TextFilter):

    ''' Tokenizes a TextStim into several word TextStims.

    Args:
        tokenizer (nltk Tokenizer or str): a nltk Tokenizer
            (or the name of one) to tokenize with. Will use
            the word_tokenize method if None is specified.
    '''

    _log_attributes = ('tokenizer',)

    def __init__(self, tokenizer=None, *args, **kwargs):
        if isinstance(tokenizer, TokenizerI):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = eval(tokenizer)(*args, **kwargs)
        else:
            self.tokenizer = None
        super(TokenizingFilter, self).__init__()

    def _filter(self, stim):
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(stim.text)
        else:
            tokens = word_tokenize(stim.text)
        stims = [TextStim(stim.filename, token, order=i)
                 for i, token in enumerate(tokens)]
        return stims


class TokenRemovalFilter(TextFilter):
    ''' Removes tokens (e.g., stopwords, common words, punctuation) from a
    TextStim.

    Args:
        tokens (list): a list of tokens (strings) to remove from a
            TextStim. Will use nltk's default stopword list if none is
            specified.
        language (str): if using the default nltk stopwords, specifies
            which language from which to use stopwords.
    '''

    _log_attributes = ('tokens', 'language')

    def __init__(self, tokens=None, language='english'):
        self.language = language
        if tokens:
            self.tokens = set(tokens)
        else:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            from nltk.corpus import stopwords
            self.tokens = set(stopwords.words(self.language))
        super(TokenRemovalFilter, self).__init__()

    def _filter(self, stim):
        tokens = word_tokenize(stim.text)
        tokens = [tok for tok in tokens if tok not in self.tokens]
        text = ' '.join(tokens)
        return TextStim(stim.filename, text)


class PunctuationRemovalFilter(TextFilter):

    ''' Removes punctuation from a TextStim. '''

    def _filter(self, stim):
        pattern = '[%s]' % re.escape(string.punctuation)
        text = re.sub(pattern, '', stim.text)
        return TextStim(stim.filename, text)


class LowerCasingFilter(TextFilter):

    ''' Lower cases the text in a TextStim. '''

    def _filter(self, stim):
        return TextStim(stim.filename, stim.text.lower())
