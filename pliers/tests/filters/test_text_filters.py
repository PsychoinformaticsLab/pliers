from os.path import join
from ..utils import get_test_data_path
from pliers.filters import (WordStemmingFilter,
                            TokenizingFilter,
                            TokenRemovalFilter,
                            LowerCasingFilter,
                            PunctuationRemovalFilter)
from pliers.graph import Graph
from pliers.stimuli import ComplexTextStim, TextStim
from nltk import stem as nls
from nltk.tokenize import PunktSentenceTokenizer
import nltk
import pytest
import string


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

    # Try a long text stim
    stim2 = TextStim(text='theres something happening here')
    filt = WordStemmingFilter()
    assert filt.transform(stim2).text == 'there someth happen here'


def test_tokenizing_filter():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'), onset=4.2)
    filt = TokenizingFilter()
    words = filt.transform(stim)
    assert len(words) == 231
    assert words[0].text == 'To'
    assert words[0].onset == 4.2
    assert words[0].order == 0
    assert words[1].onset == 4.2
    assert words[1].order == 1

    custom_tokenizer = PunktSentenceTokenizer()
    filt = TokenizingFilter(tokenizer=custom_tokenizer)
    sentences = filt.transform(stim)
    assert len(sentences) == 11
    assert sentences[0].text == 'To Sherlock Holmes she is always the woman.'

    filt = TokenizingFilter('RegexpTokenizer', '\w+|\$[\d\.]+|\S+')
    tokens = filt.transform(stim)
    assert len(tokens) == 231
    assert tokens[0].text == 'To'


def test_multiple_text_filters():
    stim = TextStim(text='testing the filtering features')
    filt1 = TokenizingFilter()
    filt2 = WordStemmingFilter()
    stemmed_tokens = filt2.transform(filt1.transform(stim))
    full_text = ' '.join([s.text for s in stemmed_tokens])
    assert full_text == 'test the filter featur'

    stim = TextStim(text='ARTICLE ONE: Rights')
    g = Graph()
    g.add_node(LowerCasingFilter())
    filt1 = LowerCasingFilter()
    filt2 = PunctuationRemovalFilter()
    filt3 = TokenizingFilter()
    final_texts = filt3.transform(filt2.transform(filt1.transform(stim)))
    assert len(final_texts) == 3
    assert final_texts[0].text == 'article'
    assert final_texts[0].order == 0
    assert final_texts[1].text == 'one'
    assert final_texts[2].text == 'rights'
    assert final_texts[2].order == 2


def test_token_removal_filter():
    stim = TextStim(text='this is not a very long sentence')
    filt = TokenRemovalFilter()
    assert filt.transform(stim).text == 'long sentence'

    filt2 = TokenRemovalFilter(tokens=['a', 'the', 'is'])
    assert filt2.transform(stim).text == 'this not very long sentence'

    stim2 = TextStim(text='More. is Real, sentence that\'ll work')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    tokens = set(stopwords.words('english')) | set(string.punctuation)
    filt3 = TokenRemovalFilter(tokens=tokens)
    assert filt3.transform(stim2).text == 'More Real sentence \'ll work'


def test_punctuation_removal_filter():
    stim = TextStim(text='this sentence, will have: punctuation, and words.')
    filt = PunctuationRemovalFilter()
    target = 'this sentence will have punctuation and words'
    assert filt.transform(stim).text == target


def test_lower_casing_filter():
    stim = TextStim(text='This is an Example Sentence.')
    filt = LowerCasingFilter()
    target = 'this is an example sentence.'
    assert filt.transform(stim).text == target
