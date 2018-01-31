from pliers.extractors import (DictionaryExtractor,
                               PartOfSpeechExtractor,
                               LengthExtractor,
                               NumUniqueWordsExtractor,
                               PredefinedDictionaryExtractor,
                               TextVectorizerExtractor,
                               WordEmbeddingExtractor,
                               VADERSentimentExtractor)
from pliers.extractors.base import merge_results
from pliers.stimuli import TextStim, ComplexTextStim
from ..utils import get_test_data_path

import numpy as np
from os.path import join
import pytest

TEXT_DIR = join(get_test_data_path(), 'text')


def test_text_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)
    result = td.transform(stim)[2].to_df()
    print(result)
    assert result['duration'][0] == 1
    assert result.shape == (1, 6)
    assert np.isclose(result['frequency'][0], 11.729, 1e-5)


def test_text_length_extractor():
    stim = TextStim(text='hello world', onset=4.2, duration=1)
    ext = LengthExtractor()
    result = ext.transform(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 11
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_unique_words_extractor():
    stim = TextStim(text='hello hello world')
    ext = NumUniqueWordsExtractor()
    result = ext.transform(stim).to_df()
    assert 'num_unique_words' in result.columns
    assert result['num_unique_words'][0] == 2


def test_dictionary_extractor():
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)

    stim = TextStim(text='annotation')
    result = td.transform(stim).to_df()
    assert np.isnan(result['onset'][0])
    assert 'length' in result.columns
    assert result['length'][0] == 10

    stim2 = TextStim(text='some')
    result = td.transform(stim2).to_df()
    assert np.isnan(result['onset'][0])
    assert 'frequency' in result.columns
    assert np.isnan(result['frequency'][0])


def test_predefined_dictionary_extractor():
    stim = TextStim(text='enormous')
    td = PredefinedDictionaryExtractor(['aoa/Freq_pm'])
    result = td.transform(stim).to_df()
    assert result.shape == (1, 5)
    assert 'aoa_Freq_pm' in result.columns
    assert np.isclose(result['aoa_Freq_pm'][0], 10.313725, 1e-5)


def test_part_of_speech_extractor():
    import nltk
    nltk.download('tagsets')
    stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
    result = merge_results(PartOfSpeechExtractor().transform(stim),
                           format='wide', extractor_names=False)
    assert result.shape == (4, 54)
    assert result['NN'].sum() == 1
    result = result.sort_values('onset')
    assert result['VBD'].iloc[3] == 1


def test_word_embedding_extractor():
    pytest.importorskip('gensim')
    stims = [TextStim(text='this'), TextStim(text='sentence')]
    ext = WordEmbeddingExtractor(join(TEXT_DIR, 'simple_vectors.bin'),
                                 binary=True)
    result = merge_results(ext.transform(stims), extractor_names='multi',
                           format='wide')
    assert ('WordEmbeddingExtractor', 'embedding_dim99') in result.columns
    assert 0.001091 in result[('WordEmbeddingExtractor', 'embedding_dim0')]


def test_vectorizer_extractor():
    pytest.importorskip('sklearn')
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    result = TextVectorizerExtractor().transform(stim).to_df()
    assert 'woman' in result.columns
    assert result['woman'][0] == 3

    from sklearn.feature_extraction.text import TfidfVectorizer
    custom_vectorizer = TfidfVectorizer()
    ext = TextVectorizerExtractor(vectorizer=custom_vectorizer)
    stim2 = TextStim(join(TEXT_DIR, 'simple_text.txt'))
    result = merge_results(ext.transform([stim, stim2]), format='wide',
                           extractor_names='multi')
    assert ('TextVectorizerExtractor', 'woman') in result.columns
    assert 0.129568189476 in result[('TextVectorizerExtractor', 'woman')]

    ext = TextVectorizerExtractor(vectorizer='CountVectorizer',
                                  analyzer='char_wb',
                                  ngram_range=(2, 2))
    result = ext.transform(stim).to_df()
    assert 'wo' in result.columns
    assert result['wo'][0] == 6


def test_vader_sentiment_extractor():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    ext = VADERSentimentExtractor()
    result = ext.transform(stim).to_df()
    assert result['sentiment_neu'][0] == 0.752

    stim2 = TextStim(text='VADER is smart, handsome, and funny!')
    result2 = ext.transform(stim2).to_df()
    assert result2['sentiment_pos'][0] == 0.752
    assert result2['sentiment_neg'][0] == 0.0
    assert result2['sentiment_neu'][0] == 0.248
    assert result2['sentiment_compound'][0] == 0.8439
