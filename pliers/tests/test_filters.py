from os.path import join
from .utils import get_test_data_path
from pliers.filters import WordStemmingFilter, TokenizingFilter
from pliers.stimuli import ComplexTextStim, TextStim
from nltk import stem as nls
from nltk.tokenize import PunktSentenceTokenizer
import pytest


TEXT_DIR = join(get_test_data_path(), 'text')


def test_word_stemming_filter():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)

    # With all defaults (porter stemmer)
    filt = WordStemmingFilter()
    assert isinstance(filt.stemmer, nls.PorterStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    target = ['some', 'sampl', 'text', 'for', 'test', 'annot']
    assert stems == target

    # Try a different stemmer
    filt = WordStemmingFilter(stemmer='snowball', language='english')
    assert isinstance(filt.stemmer, nls.SnowballStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Handles StemmerI stemmer
    stemmer = nls.SnowballStemmer(language='english')
    filt = WordStemmingFilter(stemmer=stemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Fails on invalid values
    with pytest.raises(ValueError):
        filt = WordStemmingFilter(stemmer='nonexistent_stemmer')


def test_tokenizing_filter():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    filt = TokenizingFilter()
    words = filt.transform(stim)
    assert len(words) == 231
    assert words[0].text == 'To'

    custom_tokenizer = PunktSentenceTokenizer()
    filt = TokenizingFilter(tokenizer=custom_tokenizer)
    sentences = filt.transform(stim)
    assert len(sentences) == 11
    assert sentences[0].text == 'To Sherlock Holmes she is always the woman.'


def test_multiple_text_filters():
    stim = TextStim(text='testing the filtering features')
    filt1 = TokenizingFilter()
    filt2 = WordStemmingFilter()
    stemmed_tokens = filt2.transform(filt1.transform(stim))
    full_text = ' '.join([s.text for s in stemmed_tokens])
    assert full_text == 'test the filter featur'
