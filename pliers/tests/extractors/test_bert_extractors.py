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

    # catch error if framework is invalid
    with pytest.raises(ValueError) as err:
        BertExtractor(framework='keras')
    assert 'Invalid framework' in str(err.value)

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
    res = BertExtractor(pretrained_model=model, return_input=True).transform(stim).to_df()
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
    del res, stim

 
def test_bert_sequence_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    #ext_pooler = BertSequenceEncodingExtractor(return_special='pooler_output')

    # Test correct behavior when setting return_special
    #assert ext_pooler.pooling is None
    #assert ext_pooler.return_special == 'pooler_output'

    res_sequence = BertSequenceEncodingExtractor(return_input=True).transform(stim).to_df()
    res_file =  BertSequenceEncodingExtractor(return_input=True).transform(stim_file).to_df()
    res_cls = BertSequenceEncodingExtractor(return_special='[CLS]').transform(stim).to_df()
    res_pooler = BertSequenceEncodingExtractor(return_special='pooler_output').transform(stim).to_df()
    res_max = BertSequenceEncodingExtractor(pooling='max').transform(stim).to_df()

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

    # catch error with wrong numpy function and wrong special token arg
    with pytest.raises(ValueError) as err:
        BertSequenceEncodingExtractor(pooling='avg')
    assert 'valid numpy function' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertSequenceEncodingExtractor(return_special='[MASK]')
    assert 'must be one of' in str(err.value)


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
    #ext_target = BertLMExtractor(mask=1, target=target_wds)

    res =  BertLMExtractor(mask=2).transform(stim).to_df()
    res_file =  BertLMExtractor(mask=2).transform(stim_file).to_df()
    #res_target = ext_target.transform(stim).to_df()
    res_topn = BertLMExtractor(mask=3, top_n=100).transform(stim).to_df()
    res_threshold = BertLMExtractor(mask=4, threshold=.1, return_softmax=True).transform(stim).to_df()
    res_default = BertLMExtractor().transform(stim_masked).to_df()
    res_return_mask = BertLMExtractor(mask=1, top_n=10, return_masked_word=True, return_input=True).transform(stim).to_df()

    assert res.shape[0] == 1

    # test onset/duration
    assert res_file['onset'][0] == 1.0
    assert res_file['duration'][0] == 0.2

    # Check target words
    #assert all([w.capitalize() in res_target.columns for w in target_wds])
    #assert res_target.shape[1] == 6

    # Check top_n
    assert res_topn.shape[1] == 104
    assert all([res_topn.iloc[:,3][0] > res_topn.iloc[:,i][0] for i in range(4,103)])

    # Check threshold and return_softmax
    tknz = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tknz.vocab.keys()
    for v in vocab:
        if v.capitalize() in res_threshold.columns:
            assert res_threshold[v.capitalize()][0] >= .1
            assert res_threshold[v.capitalize()][0] <= 1

    # Test update mask method
    #assert ext_target.mask == 1
    #ext_target.update_mask(new_mask='sentence')
    #assert ext_target.mask == 'sentence'
    #res_target_new = ext_target.transform(stim).to_df()
    #assert all([res_target[c][0] != res_target_new[c][0]
    #            for c in ['Target', 'Word']])
    #with pytest.raises(ValueError) as err:
    #    ext_target.update_mask(new_mask=['some', 'mask'])
    #assert 'must be a string' in str(err.value)
    
    # Test default mask
    assert res_default.shape[0] == 1

    # Test return mask and input
    assert res_return_mask['true_word'][0] == 'is'
    assert 'true_word_score' in res_return_mask.columns
    assert res_return_mask['sequence'][0] == 'This is not a tokenized sentence .'


def test_bert_sentiment_extractor():
    stim = ComplexTextStim(text='This is the best day of my life.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    res = BertSentimentExtractor().transform(stim).to_df()
    res_file = BertSentimentExtractor().transform(stim_file).to_df()
    res_seq = BertSentimentExtractor(return_input=True).transform(stim).to_df()
    res_softmax = BertSentimentExtractor(return_softmax=True).transform(stim).to_df()

    assert res.shape[0] == 1
    assert res_file['onset'][0] == 0.2
    assert res_file['duration'][0] == 2.9
    assert all([s in res.columns for s in ['sent_pos', 'sent_neg']])
    assert res_seq['sequence'][0] == 'This is the best day of my life .'
    assert all([res_softmax[s][0] >= 0 for s in ['sent_pos','sent_neg'] ])
    assert all([res_softmax[s][0] <= 1 for s in ['sent_pos','sent_neg'] ])

