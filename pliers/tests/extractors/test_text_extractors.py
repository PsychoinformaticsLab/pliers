from pliers import config
from pliers.extractors import (DictionaryExtractor,
                               PartOfSpeechExtractor,
                               LengthExtractor,
                               NumUniqueWordsExtractor,
                               PredefinedDictionaryExtractor,
                               TextVectorizerExtractor,
                               WordEmbeddingExtractor,
                               VADERSentimentExtractor,
                               SpaCyExtractor,
                               BertExtractor,
                               BertSequenceEncodingExtractor,
                               BertLMExtractor,
                               BertSentimentExtractor,
                               WordCounterExtractor)
from pliers.extractors.base import merge_results
from pliers.stimuli import TextStim, ComplexTextStim
from pliers.tests.utils import get_test_data_path
import numpy as np
from os.path import join
from pathlib import Path
import shutil
import pytest
import spacy
from os import environ
from transformers import BertTokenizer

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
    assert result['pos_'][1] == 'AUX'
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


def test_bert_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))
    
    ext_base = BertExtractor(pretrained_model='bert-base-uncased')
    ext_base_token = BertExtractor(pretrained_model='bert-base-uncased',
                                   return_input=True)
    
    base_result = ext_base.transform(stim)
    res = base_result.to_df()
    res_model_attr = base_result.to_df(include_attributes=True)
    res_token = ext_base_token.transform(stim).to_df()
    res_file = ext_base.transform(stim_file).to_df()
    
    # Test encoding shape
    assert len(res['encoding'][0]) == 768
    assert len(res_file['encoding'][0]) == 768

    # test base extractor
    assert res.shape[0] == 8
    assert res_token.shape[0] == 8
    assert res_token['token'][5] == '##ized'
    assert res_token['word'][5] == 'tokenized'
    assert res_token['object_id'][5] == 5
    
    # test base extractor on file
    assert res_file.shape[0] == 8
    assert res_file['onset'][3] == 1.3
    assert res_file['duration'][5] == 0.5
    assert res_file['object_id'][5] == 5

    # test model attributes
    assert all([a in res_model_attr.columns for a in ext_base._model_attributes])

    # catch error if framework is invalid
    with pytest.raises(ValueError) as err:
        BertExtractor(framework='keras')
    assert 'Invalid framework' in str(err.value)

    # delete the model
    home = Path.home()
    model_path = str(home / '.cache' / 'torch' / 'transformers')
    shutil.rmtree(model_path)

    # Delete the models
    del res, res_token, res_file, ext_base, ext_base_token


@pytest.mark.parametrize('model', ['bert-large-uncased', 
                                   'distilbert-base-uncased',
                                   'roberta-base','camembert-base'])
def test_bert_other_models(model):
    if model == 'camembert-base':
        stim = ComplexTextStim(text='ceci n\'est pas un pipe')
    else:
        stim = ComplexTextStim(text='This is not a tokenized sentence.')
    ext = BertExtractor(pretrained_model=model, return_input=True)
    res = ext.transform(stim).to_df()
    if model == 'bert-large-uncased':
        shape = 1024
    else:
        shape = 768
    assert len(res['encoding'][0]) == shape
    if model == 'camembert-base':
        assert res['token'][4] == 'est'

    # delete the model
    home = Path.home()
    model_path = str(home / '.cache' / 'torch' / 'transformers')
    shutil.rmtree(model_path)

    # remove variables
    del ext, res, stim

''' 
def test_bert_sequence_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    ext_sequence = BertSequenceEncodingExtractor(return_input=True)
    print('Initialized ext_seq')
    ext_cls = BertSequenceEncodingExtractor(return_special='[CLS]')
    print('Initialized ext_cls')
    ext_pooler = BertSequenceEncodingExtractor(return_special='pooler_output')
    print('Initialized ext_pooler')
    ext_max = BertSequenceEncodingExtractor(pooling='max')
    print('Initialized ext_max')

    # Test correct behavior when setting return_special
    assert ext_cls.pooling is None
    assert ext_pooler.pooling is None
    assert ext_cls.return_special == '[CLS]'
    assert ext_pooler.return_special == 'pooler_output'

    res_sequence = ext_sequence.transform(stim).to_df()
    res_file = ext.transform(stim_file).to_df()
    res_cls = ext_cls.transform(stim).to_df()
    res_pooler = ext_pooler.transform(stim).to_df()
    res_max = ext_max.transform(stim).to_df()

    # Check shape
    assert len(res_sequence['encoding'][0]) == 768
    assert len(res_cls['encoding'][0]) == 768
    assert len(res_pooler['encoding'][0]) == 768
    assert len(res_max['encoding'][0]) == 768
    assert res_sequence.shape[0] == 1
    assert res_cls.shape[0] == 1
    assert res_pooler.shape[0] == 1
    assert res_max.shape[0] == 1

    # Make sure pooler/cls/no arguments return different encodings
    assert res_sequence['encoding'][0] != res_cls['encoding'][0]
    assert res_sequence['encoding'][0] != res_pooler['encoding'][0]
    assert res_sequence['encoding'][0] != res_max['encoding'][0]
    assert all([res_max['encoding'][0][i] >= res_sequence['encoding'][0][i]
                                              for i in range(768)])

    # test return sequence
    assert res_sequence['sequence'][0] == 'This is not a tokenized sentence .'

    # test file stim
    assert res_file['duration'][0] == 2.9
    assert res_file['onset'][0] == 0.2

    # test tf vs. torch
    assert np.isclose(cor, 1)

    # catch error with wrong numpy function and wrong special token arg
    with pytest.raises(ValueError) as err:
        BertSequenceEncodingExtractor(pooling='avg')
    assert 'valid numpy function' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertSequenceEncodingExtractor(return_special='[MASK]')
    assert 'must be one of' in str(err.value)

    # delete the model
    home = Path.home()
    model_path = str(home / '.cache' / 'torch' / 'transformers')
    shutil.rmtree(model_path)

    del ext, ext_sequence, ext_cls, ext_pooler, ext_max


def test_bert_LM_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_masked = ComplexTextStim(text='This is MASK tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    # Test mutual exclusivity and mask values
    with pytest.raises(ValueError) as err:
        BertLMExtractor(top_n=100, target='test')
    assert 'mutually exclusive' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(top_n=100, threshold=.5)
    assert 'mutually exclusive' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(target='test', threshold=.5)
    assert 'mutually exclusive' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(mask=['test', 'mask'])
    assert 'must be a string' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(target='nonwd')
    assert 'No valid target token' in str(err.value)

    target_wds = ['target','word']
    ext = BertLMExtractor(mask=2)
    ext_masked = BertLMExtractor()
    ext_target = BertLMExtractor(mask=1, target=target_wds)
    ext_topn = BertLMExtractor(mask=3, top_n=100)
    ext_threshold = BertLMExtractor(mask=4, threshold=.1, return_softmax=True)
    ext_default = BertLMExtractor()
    ext_return_mask = BertLMExtractor(mask=1, top_n=10, 
                                      return_masked_word=True, return_input=True)

    res = ext.transform(stim).to_df()
    res_masked = ext_masked.transform(stim_masked).to_df()
    res_file = ext.transform(stim_file).to_df()
    res_target = ext_target.transform(stim).to_df()
    res_topn = ext_topn.transform(stim).to_df()
    res_threshold = ext_threshold.transform(stim).to_df()
    res_default = ext_default.transform(stim_masked).to_df()
    res_return_mask = ext_return_mask.transform(stim).to_df()

    assert res.shape[0] == 1

    # test onset/duration
    assert res_file['onset'][0] == 1.0
    assert res_file['duration'][0] == 0.2

    # Check target words
    assert all([w.capitalize() in res_target.columns for w in target_wds])
    assert res_target.shape[1] == 13

    # Check top_n
    assert res_topn.shape[1] == 111
    assert all([res_topn.iloc[:,3][0] > res_topn.iloc[:,i][0] for i in range(4,103)])

    # Check threshold and return_softmax
    tknz = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tknz.vocab.keys()
    for v in vocab:
        if v.capitalize() in res_threshold.columns:
            assert res_threshold[v.capitalize()][0] >= .1
            assert res_threshold[v.capitalize()][0] <= 1

    # Test update mask method
    assert ext_target.mask == 1
    ext_target.update_mask(new_mask='sentence')
    assert ext_target.mask == 'sentence'
    res_target_new = ext_target.transform(stim).to_df()
    assert all([res_target[c][0] != res_target_new[c][0]
                for c in ['Target', 'Word', 'mask']])
    with pytest.raises(ValueError) as err:
        ext_target.update_mask(new_mask=['some', 'mask'])
    assert 'must be a string' in str(err.value)
    
    # Test default mask
    assert res_default.shape[0] == 1
    assert res_default['mask'][0] == 'MASK'

    # Test return mask and input
    assert res_return_mask['true_word'][0] == 'is'
    assert 'true_word_score' in res_return_mask.columns
    assert res_return_mask['sequence'][0] == 'This is not a tokenized sentence .'

    # delete the model
    home = Path.home()
    model_path = str(home / '.cache' / 'torch' / 'transformers')
    shutil.rmtree(model_path)

    # remove
    del ext, ext_masked, ext_target, ext_topn, ext_threshold, ext_default, \
        ext_return_mask
    del res, res_masked, res_file, res_target, res_topn, res_threshold, \
        res_default, res_return_mask

def test_bert_sentiment_extractor():
    stim = ComplexTextStim(text='This is the best day of my life.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    ext = BertSentimentExtractor()
    ext_seq = BertSentimentExtractor(return_input=True)
    ext_softmax = BertSentimentExtractor(return_softmax=True)

    res = ext.transform(stim).to_df()
    res_file = ext.transform(stim_file).to_df()
    res_seq = ext_seq.transform(stim).to_df()
    res_softmax = ext_softmax.transform(stim).to_df()

    assert res.shape[0] == 1
    assert res_file['onset'][0] == 0.2
    assert res_file['duration'][0] == 2.9
    assert all([s in res.columns for s in ['sent_pos', 'sent_neg']])
    assert res_seq['sequence'][0] == 'This is the best day of my life .'
    assert all([res_softmax[s][0] >= 0 for s in ['sent_pos','sent_neg'] ])
    assert all([res_softmax[s][0] <= 1 for s in ['sent_pos','sent_neg'] ])

    # delete the model
    home = Path.home()
    model_path = str(home / '.cache' / 'torch' / 'transformers')
    shutil.rmtree(model_path)

    del ext, ext_seq, ext_softmax
    del res, res_file, res_seq, res_softmax
'''

def test_word_counter_extractor():
    stim_txt = ComplexTextStim(text='This is a text where certain words occur'
                                    ' again and again Sometimes they are '
                                    'lowercase sometimes they are uppercase '
                                    'There are also words that may look '
                                    'different but they come from the same '
                                    'lemma Take a word like text and its '
                                    'plural texts Oh words')
    stim_with_onsets = ComplexTextStim(filename=join(TEXT_DIR,
                                       'complex_stim_with_repetitions.txt'))
    ext = WordCounterExtractor()
    result_stim_txt = ext.transform(stim_txt).to_df()
    result_stim_with_onsets = ext.transform(stim_with_onsets).to_df()
    assert result_stim_txt.shape[0] == 45
    assert all(result_stim_txt['word_count'] >= 1)
    assert result_stim_txt['word_count'][15] == 2
    assert result_stim_txt['word_count'][44] == 3
    assert result_stim_with_onsets.shape[0] == 8
    assert result_stim_with_onsets['onset'][2] == 0.8
    assert result_stim_with_onsets['duration'][2] == 0.1
    assert result_stim_with_onsets['word_count'][2] == 2
    assert result_stim_with_onsets['word_count'][5] == 2
    assert result_stim_with_onsets['word_count'][7] == 1

    ext2 = WordCounterExtractor(log_scale=True)
    result_stim_txt = ext2.transform(stim_txt).to_df()
    assert all(result_stim_txt['log_word_count'] >= 0)
    assert result_stim_txt['log_word_count'][15] == np.log(2)
    assert result_stim_txt['log_word_count'][44] == np.log(3)

