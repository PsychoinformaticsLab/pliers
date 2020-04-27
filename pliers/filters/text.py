''' Filters that operate on TextStim inputs. '''

import string
import re

import nltk
from nltk import stem
from nltk.tokenize import word_tokenize
from nltk.tokenize import * # noqa
from nltk.tokenize.api import TokenizerI

from pliers.stimuli.text import TextStim
from pliers.support.decorators import requires_nltk_corpus
from .base import Filter


class TextFilter(Filter):

    ''' Base class for all TextFilters. '''

    _input_type = TextStim


class WordStemmingFilter(TextFilter):

    ''' Nltk-based word stemming and lemmatization Filter.

    Args:
        stemmer (str, Stemmer): If a string, must be the name of one of the
            stemming and lemmatization modules available in nltk.stem.
            Valid values are 'porter', 'snowball', 'isri', 'lancaster',
            'regexp', 'wordnet', or 'rslp'. Alternatively, an initialized
            nltk StemmerI instance can be passed.
        tokenize (bool): if True, tokenize using nltk.word_tokenize and apply
            stemmer/lemmatizer to each token. If False, do not tokenize before
            stemming/lemmatizing.
        case_sensitive (bool): if False (default), input is lower-cased before
            stemming or lemmatizing.
        args, kwargs: Optional positional and keyword args passed onto the
            nltk stemmer/lemmatizer.
    '''

    _stemmers = {
        'porter': 'PorterStemmer',
        'snowball': 'SnowballStemmer',
        'lancaster': 'LancasterStemmer',
        'isri': 'ISRIStemmer',
        'regexp': 'RegexpStemmer',
        'rslp': 'RSLPStemmer',
        'wordnet': 'WordNetLemmatizer'
    }

    _log_attributes = ('stemmer', 'tokenize', 'case_sensitive')

    @requires_nltk_corpus
    def __init__(self, stemmer='porter', tokenize=True, case_sensitive=False,
                 *args, **kwargs):
        if isinstance(stemmer, str):
            if stemmer not in self._stemmers:
                valid = list(self._stemmers.keys())
                raise ValueError("Invalid stemmer '%s'; please use one of %s."
                                 % (stemmer, valid))
            stemmer = getattr(stem, self._stemmers[stemmer])(*args, **kwargs)
        elif not isinstance(stemmer, (stem.StemmerI, stem.WordNetLemmatizer)):
            raise ValueError("stemmer must be either a valid string, or an "
                             "instance of class StemmerI.")
        self.stemmer = stemmer
        self.tokenize = tokenize
        self.case_sensitive = case_sensitive
        super(WordStemmingFilter, self).__init__()

    @requires_nltk_corpus
    def _filter(self, stim):
        pos_map = {
            'ADJ': 'a',
            'ADJ_SAT': 's',
            'ADV': 'r',
            'NOUN': 'n',
            'VERB': 'v'
        }

        def pos_wordnet(txt):
            pos_tagged = dict(nltk.pos_tag(txt, tagset='universal'))
            pos_tagged = {t: pos_map[tag] if tag in pos_map else 'n'
                          for t, tag in pos_tagged.items()}
            return pos_tagged

        tokens = [stim.text]
        if self.tokenize:
            tokens = nltk.word_tokenize(tokens[0])
        tokens = [t if self.case_sensitive else t.lower() for t in tokens]
        if not isinstance(self.stemmer, stem.WordNetLemmatizer):
            stemmed = ' '.join([self.stemmer.stem(t) for t in tokens])
        else:
            pos_tagged = pos_wordnet(tokens)
            stemmed = ' '.join([self.stemmer.lemmatize(t, pos=pos_tagged[t])
                                for t in tokens])
        return TextStim(stim.filename, stemmed, stim.onset, stim.duration,
                        stim.order, stim.url)


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
