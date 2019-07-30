from pliers.extractors import (DictionaryExtractor,
                               PartOfSpeechExtractor,
                               LengthExtractor,
                               NumUniqueWordsExtractor,
                               PredefinedDictionaryExtractor,
                               TextVectorizerExtractor,
                               WordEmbeddingExtractor,
                               VADERSentimentExtractor)

from pliers.extractors import DirectSentenceExtractor
from pliers.extractors import DirectTextExtractorInterface


from pliers.extractors.base import merge_results
from pliers.stimuli import TextStim, ComplexTextStim
from pliers.tests.utils import get_test_data_path

import numpy as np
from os.path import join
import pytest

TEXT_DIR = join(get_test_data_path(), 'text')

def test_doc2vec_extractors():
    
    print('doc2vec test')
    
    pytest.importorskip('gensim')
    rep = 'doc2vec'
    ext = DirectTextExtractorInterface(method=rep)
    input = ['Colorless green ideas sleep furiously .']
    
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 300

    '''
    Note: doc2vec has an inherent randomnes in the
    initialization part of the algorithm. So the following
    assert may fail
    assert np.isclose(vector[0][11],-0.021118578,1e-20)
    assert np.isclose(vector[0][36],0.48503822,1e-20)
    '''


def test_bert_extractors():
    
    print('embedding test')
    #testing skipthought 
                           
    pytest.importorskip('gensim')
    rep = 'bert'
    ext = DirectTextExtractorInterface(method=rep)
    input = ['Colorless green ideas sleep furiously .']
    
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 768

    assert np.isclose(vector[0][11],0.006889418154822454,1e-10)
    assert np.isclose(vector[0][36],0.012296270387437608,1e-10)

def test_sif_extractors():
    
    print('embedding test')
    #testing skipthought 
                           
    pytest.importorskip('gensim')
    rep = 'sif'
    ext = DirectTextExtractorInterface(method=rep)
    input = ['Colorless green ideas sleep furiously .']
    
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 300
    assert np.isclose(vector[0][11],0.006889418154822454,1e-10)
    assert np.isclose(vector[0][36],0.012296270387437608,1e-10)


def test_elmo_extractors():
    
    print('embedding test')
    #testing skipthought 
                           
    pytest.importorskip('gensim')
    rep = 'elmo'
    ext = DirectTextExtractorInterface(method=rep)
    input = ['Colorless green ideas sleep furiously .']
    
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 512

    assert np.isclose(vector[0][11],0.15085818,1e-10)
    assert np.isclose(vector[0][36],-0.1827029,1e-10)


def test_skipthought_extractors():
    
    # Note: skipthought is based on Theano 1.0.4 and
    # Numpy <1.16.3 
    # To run the following test case please 
    # change the Theano and Numpy versions

    print('embedding test')
    #testing skipthought 
                           
    pytest.importorskip('gensim')
    rep = 'skipthought'
    ext = DirectTextExtractorInterface(method=rep)
    input = ['Colorless green ideas sleep furiously .']
    
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 786
'''
def test_average_word_embedding_extractors():
    
    print('embedding test')
    #we test three types of word embeddings
     #   (glove, word2vec, fasttext)
    
    pytest.importorskip('gensim')
    rep = 'averagewordembedding'
    ext = DirectTextExtractorInterface(method=rep,embedding='glove',dimensionality=50,content_only=True)
    input = ['Colorless green ideas sleep furiously .']
    
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 50
    
    assert np.isclose(vector[0][11],0.061906997859477994,1e-10)
    assert np.isclose(vector[0][36],-0.2516959875822067,1e-10)
    
    ext = DirectTextExtractorInterface(method=rep,embedding='word2vec')
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 300
    
    assert np.isclose(vector[0][11],-0.030078199878335,1e-10)
    assert np.isclose(vector[0][36],0.02783220112323761,1e-10)


    ext = DirectTextExtractorInterface(method=rep,embedding='fasttext')
    result = ext.embed(input)
    vector = result._data
    stim = result.stim
    
    assert len(vector) == 1
    assert len(vector[0]) == 300
    
    assert np.isclose(vector[0][11],0.24355999901890754,1e-10)
    assert np.isclose(vector[0][36],0.03787999898195267,1e-10)



