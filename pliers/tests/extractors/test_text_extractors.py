from pliers.extractors import (DictionaryExtractor,
                               PartOfSpeechExtractor,
                               LengthExtractor,
                               NumUniqueWordsExtractor,
                               PredefinedDictionaryExtractor,
                               TextVectorizerExtractor,
                               WordEmbeddingExtractor,
                               VADERSentimentExtractor,
                               SpaCyExtractor)
from pliers.extractors.base import merge_results
from pliers.stimuli import TextStim, ComplexTextStim
from ..utils import get_test_data_path

import numpy as np
from os.path import join
import pytest
import spacy

TEXT_DIR = join(get_test_data_path(), 'text')


def test_text_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)
    result = td.transform(stim)[2].to_df()
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


def test_predefined_dictionary_retrieval():
    variables = [
        'affect/D.Mean.H',
        'concreteness/SUBTLEX',
        'subtlexusfrequency/Zipf-value',
        'calgarysemanticdecision/RTclean_mean',
        'massiveauditorylexicaldecision/PhonLev'
    ]
    stim = TextStim(text='perhaps')
    td = PredefinedDictionaryExtractor(variables)
    result = td.transform(stim).to_df().iloc[0]
    assert np.isnan(result['affect_D.Mean.H'])
    assert result['concreteness_SUBTLEX'] == 6939
    assert result['calgarysemanticdecision_RTclean_mean'] == 954.48
    assert np.isclose(result['subtlexusfrequency_Zipf-value'], 5.1331936)
    assert np.isclose(result['massiveauditorylexicaldecision_PhonLev'],
                      6.65101626)


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
    assert np.allclose(0.0010911,
                       result[('WordEmbeddingExtractor', 'embedding_dim0')][0])

    unk = TextStim(text='nowaythisinvocab')
    result = ext.transform(unk).to_df()
    assert result['embedding_dim10'][0] == 0.0

    ones = np.ones(100)
    ext = WordEmbeddingExtractor(join(TEXT_DIR, 'simple_vectors.bin'),
                                 binary=True, unk_vector=ones)
    result = ext.transform(unk).to_df()
    assert result['embedding_dim10'][0] == 1.0

    ext = WordEmbeddingExtractor(join(TEXT_DIR, 'simple_vectors.bin'),
                                 binary=True, unk_vector='random')
    result = ext.transform(unk).to_df()
    assert result['embedding_dim10'][0] <= 1.0
    assert result['embedding_dim10'][0] >= -1.0

    ext = WordEmbeddingExtractor(join(TEXT_DIR, 'simple_vectors.bin'),
                                 binary=True, unk_vector='nothing')
    result = ext.transform(unk).to_df()
    assert result['embedding_dim10'][0] == 0.0


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
    assert np.allclose(0.129568189476,
                       result[('TextVectorizerExtractor', 'woman')][0])

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


def test_spacy_token_extractor():
    pytest.importorskip('spacy')
    stim = TextStim(text='This is a test.')
    ext = SpaCyExtractor(extractor_type='token')
    assert ext.model is not None

    ext2 = SpaCyExtractor(model='en_core_web_sm')
    assert isinstance(ext2.model, spacy.lang.en.English)

    result = ext.transform(stim).to_df()
    assert result['text'][0] == 'This'
    assert result['lemma_'][0].lower() == 'this'
    assert result['pos_'][0] == 'DET'
    assert result['tag_'][0] == 'DT'
    assert result['dep_'][0] == 'nsubj'
    assert result['shape_'][0] == 'Xxxx'
    assert result['is_alpha'][0] == 'True'
    assert result['is_stop'][0] == 'True'
    assert result['is_punct'][0] == 'False'
    assert result['is_ascii'][0] == 'True'
    assert result['is_digit'][0] == 'False'
    assert result['sentiment'][0] == '0.0'

    assert result['text'][1] == 'is'
    assert result['lemma_'][1].lower() == 'be'
    assert result['pos_'][1] == 'VERB'
    assert result['tag_'][1] == 'VBZ'
    assert result['dep_'][1] == 'ROOT'
    assert result['shape_'][1] == 'xx'
    assert result['is_alpha'][1] == 'True'
    assert result['is_stop'][1] == 'True'
    assert result['is_punct'][1] == 'False'
    assert result['is_ascii'][1] == 'True'
    assert result['is_digit'][1] == 'False'
    assert result['sentiment'][1] == '0.0'

    assert result['text'][2] == 'a'
    assert result['lemma_'][2].lower() == 'a'
    assert result['pos_'][2] == 'DET'
    assert result['tag_'][2] == 'DT'
    assert result['dep_'][2] == 'det'
    assert result['shape_'][2] == 'x'
    assert result['is_alpha'][2] == 'True'
    assert result['is_stop'][2] == 'True'
    assert result['is_punct'][2] == 'False'
    assert result['is_ascii'][2] == 'True'
    assert result['is_digit'][2] == 'False'
    assert result['sentiment'][2] == '0.0'

    assert result['text'][3] == 'test'
    assert result['lemma_'][3].lower() == 'test'
    assert result['pos_'][3] == 'NOUN'
    assert result['tag_'][3] == 'NN'
    assert result['dep_'][3] == 'attr'
    assert result['shape_'][3] == 'xxxx'
    assert result['is_alpha'][3] == 'True'
    assert result['is_stop'][3] == 'False'
    assert result['is_punct'][3] == 'False'
    assert result['is_ascii'][3] == 'True'
    assert result['is_digit'][3] == 'False'
    assert result['sentiment'][3] == '0.0'


def test_spacy_doc_extractor():
    pytest.importorskip('spacy')
    stim2 = TextStim(text='This is a test. And we are testing again. This '
                     'should be quite interesting. Tests are totally fun.')
    ext = SpaCyExtractor(extractor_type='doc')
    assert ext.model is not None

    result = ext.transform(stim2).to_df()
    assert result['text'][0]=='This is a test. '
    assert result['is_parsed'][0]
    assert result['is_tagged'][0]
    assert result['is_sentenced'][0]

    assert result['text'][3]=='Tests are totally fun.'
    assert result['is_parsed'][3]
    assert result['is_tagged'][3]
    assert result['is_sentenced'][3]
