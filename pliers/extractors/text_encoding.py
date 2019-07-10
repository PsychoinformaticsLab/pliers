'''
New Extractors that operate exclusively on Text stimuli and
returns embedding via different SOTA models,
such as: word2vec, glove, fasttext (average embedding)
       : skipthought, doc2vec, ELMO, BERT, Deep Average n/w.
       
Debanjan Ghosh
Francisco Pereira

'''
from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.support.exceptions import PliersError
from pliers.support.decorators import requires_nltk_corpus
from pliers.datasets.text import fetch_dictionary
from pliers.transformers import BatchTransformerMixin
from pliers.utils import attempt_to_import, verify_dependencies
from pliers.extractors.text import TextExtractor

import gensim
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

import os
import numpy as np
import pandas as pd
import sys
from six import string_types
import gzip
from enum import Enum   

'''uncomment the following two imports (tf and hub)
   to run DAN'''
#import tensorflow as tf
#import tensorflow_hub as hub 


'''bert related helper code'''
from pliers.extractors import bert_modeling
from pliers.extractors import bert_tokenization
from pliers.extractors import bert_extract_features

'''skipthought related helper code '''
from pliers.extractors import skipthoughts

'''SIF related helper code'''
from pliers.extractors import sif_data_io,sif_params,SIF_embedding

''' import elmo related code from AllenAI'''
from allennlp.commands.elmo import ElmoEmbedder


embedding_methods = Enum('embedding_methods', 'average_embedding word2vec glove')

keyedvectors = attempt_to_import('gensim.models.keyedvectors', 'keyedvectors',
                                 ['KeyedVectors'])
doc2vecVectors = attempt_to_import('gensim.models.doc2vec','doc2vecVectors',
                                   ['Doc2Vec.load'])

class DirectTextExtractorInterface():
    
    '''
        Args:
        method (str): The name of the embedding methods. The possibilities
        (averageembedding, doc2vec, sif...) will be provided to the users
        via a README.
        (default:averageembedding) 
        
        Embedding = “glove|word2vec|dep2vec|fasttext”  
        (default:glove)
        
        Dimensionality = “50|100|200|300” [for glove]; 300 
        [for word2vec and dep2vec] (default:50)

        corpus=”42B,” [for glove; default is 42B]. 
        Note, unlike glove, the other models do not come up with so many choices. See README
        for particular choice of corpus and diemensionality 

        content_only = True|False [default:Content_only = True]
        
        binary = True|False [vector in binary or not, applicable for averageembedding]
        
        stopWords= None| list [default: NLTK stop words]

        unk_vector (numpy array or str): Default vector to use for texts not
            found in the embedding file. If None is specified, uses a
            vector with all zeros. If 'random' is specified, uses a vector with
            random values between -1.0 and 1.0. Must have the same dimensions
            as the embeddings.
            
        layer (str description): For models such as ELMo user can select
            a specific layer for encoding. This variable gives the user
            the option of selecting a particular layer. Default (i.e., 
            not specifying any value) is based on the model's specificity.
    '''
    
    def __init__(self,method="averageWordEmbedding",\
                             embedding="glove",\
                             dimensionality=300 ,\
                             corpus="840B",\
                             content_only=True,\
                             binary=False,\
                             stopWords=None,\
                             unk_vector=None,\
                             layer=None):
        
        verify_dependencies(['keyedvectors']) 
        verify_dependencies(['doc2vecVectors']) 
        
        '''Check the instance type of all the parameters'''
        
        if not isinstance(method,str):
            raise ValueError('Method should be string (default is ' + \
            'AverageWordEmbedding) or select' + \
            ' from the list provided in README.')
         
        '''checking instance type of the parameters is done'''
                
        self.method = method
        if self.method.lower() == "averagewordembedding":
            
            self.semvector_object = AverageEmbeddingExtractor(embedding=embedding,\
                             dimensionality=dimensionality,\
                             corpus=corpus,\
                             content_only=content_only,\
                             binary = binary,\
                             stopWords=stopWords)
            
            self.semvector_object._loadModel()


        elif method.lower() == 'skipthought':
            
            self.semvector_object = SkipThoughtExtractor()
            self.semvector_object._loadModel() 
            
        elif method.lower() == 'sif':
            
            self.semvector_object = SmoothInverseFrequencyExtractor(embedding=embedding,\
                             dimensionality=dimensionality,\
                             corpus=corpus,\
                             content_only=content_only,\
                             stopWords=stopWords)
            self.semvector_object._loadModel()
            
        elif method.lower() == 'doc2vec':
            
            self.semvector_object = Doc2vecExtractor()
            self.semvector_object._loadModel()
        
        elif method.lower() == 'elmo':
            self.semvector_object = ElmoExtractor(layer=layer)
            
        elif method.lower() == 'dan':
            self.semvector_object = DANExtractor()
            
        elif method.lower() == 'bert':
            self.semvector_object = BertExtractor()
        
        else:
            raise ValueError('Method: ' + '\"' + method + '\"' ' is not supported. Default is ' + \
            'AverageWordEmbedding or select' + \
            ' from the list provided in README.')
            
            
        super(DirectTextExtractorInterface, self).__init__()
        
    def _embed(self,stim):
        
        ''' 
            we need to know specific method and
            embedding that has been chosen
            
        '''
        if self.method.lower() in [ "averagewordembedding", \
                                  "skipthought", \
                                  "sif", \
                                  "doc2vec", \
                                  "dan",\
                                  "elmo",\
                                  "bert" ] :
        
            return self.semvector_object._embed(stim)

class DirectSentenceExtractor(TextExtractor):

    ''' 
        A extractor (parent class) that uses a word or sentence embedding 
        to generating embedding for text (any).
        Note, this is different from the current WordEmbedding
        Extractor in the text.py file in Pliers. Eventually this will replace
        the current WordEmbeddingExtractor.
    '''
    
    _log_attributes = ('wvModel', 'prefix')
    _available_word_embeddings = ['glove', 'word2vec','context2vec', 'fasttext']
    _version = '0.1'
    prefix = 'embedding_dim'
    
    _vectors = 'vectors'
    _text = '.txt'
    
    def __init__(self,method="averageWordEmbedding",\
                             embedding="glove",\
                             dimensionality=300,\
                             corpus="6B",\
                             content_only=True,\
                             stopWords=None,\
                             unk_vector=None):
        
        verify_dependencies(['keyedvectors']) 
        verify_dependencies(['doc2vecVectors']) 
        self._unk_vector = unk_vector
        
        self._current_path = os.path.abspath(os.path.dirname(__file__))
        self._embedding_model_path = os.path.join(self._current_path,'../../datasets/embeddings/')


        super(DirectSentenceExtractor, self).__init__()
       
    def _extract(self, stim):
        num_dims = self.wvModel.vector_size
        if stim.text in self.wvModel:
            embedding_vector = self.wvModel[stim.text]
        else:
            unk = self._unk_vector
            if hasattr(unk, 'shape') and unk.shape[0] == num_dims:
                embedding_vector = unk
            elif unk == 'random':
                embedding_vector = 2.0 * np.random.random(num_dims) - 1.0
            else:
                # By default, UNKs will have zeroed-out vectors
                embedding_vector = np.zeros(num_dims)

        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        return ExtractorResult([embedding_vector],
                               stim,
                               self,
                               features=features)

class SmoothInverseFrequencyExtractor(DirectSentenceExtractor):
    
    '''Note, all parameters are default and as used in the 
        Paper. For more information see:
        A Simple But Tough-To-Beat Baseline For Sentence
        Embeddings by Arora et al. ICLR 2017.
        The code is adapted from https://github.com/PrincetonML/SIF'''

    def __init__(self,embedding='glove',dimensionality=300,\
               content_only=True,stopWords=None,corpus='840B',unk_vector=None):        

        print('inside smooth inverse frequency extractor')
        super(SmoothInverseFrequencyExtractor, self).__init__()

        
        self._method = 'sif'
        self._weightFile = 'enwiki_vocab_min200.txt'
        self._weightFile = os.path.join(self._embedding_model_path,self._method,self._weightFile)
        
        '''We use the default embedding (glove) as 
           used in the original work. So other parameterds
           do not have any effect'''
        self.embedding = 'glove'
        self.dimensionality = 50
        self.corpus = '6B'
        
        ''' Check whether the embedding file exist '''
        
        embedding_input =  self.embedding + self.corpus  + str(self.dimensionality)  + self._text
        self._sif_model_path = os.path.join(self._embedding_model_path,self.embedding)
        self.embedding_file = os.path.join(self._sif_model_path, embedding_input)

        if not os.path.exists(self.embedding_file):
            raise ValueError('Embedding model file ' + \
                                'is missing. Please download ' + \
                                'the model files via the following command: ' +\
                                'python download.py \'[embedding_name]\', where [embedding_name] ' +\
                                'is from [glove,word2vec,fasttext,dep2vec]')

        if not os.path.exists(self._weightFile):
            raise ValueError('sif model file ' + self._weightFile + 
                                 ' '\
                                ' is missing. Please download ' + \
                                ' the model files via the following command: ' +\
                                ' python download.py (\'sif\')' )

        
    def _loadModel(self):
        
        self.weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
        self.rmpc = 1 # number of principal components to remove in SIF weighting scheme
        # load word vectors
        (self.word_vecs, self.We) = sif_data_io.getWordmap(self.embedding_file,self.dimensionality)
            # load word weights
        word2weight = sif_data_io.getWordWeight(self._weightFile, self.weightpara) # word2weight['str'] is the weight for the word 'str'
        self.weight4ind = sif_data_io.getWeight(self.word_vecs, word2weight) # weight4ind[i] is the weight for the i-th word

    def setEmbeddingFile(self,embedding_file):
        
        self.embedding_file = embedding_file
        
    def _embed(self,stim):
        
        if not isinstance(stim,list):
            stim = [stim]
        
        x, m = sif_data_io.sentences2idx(stim, self.word_vecs) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = sif_data_io.seq2weight(x, m, self.weight4ind) # get word weights
            
            # set parameters
        params = sif_params.sif_params()
        params.rmpc = self.rmpc
            # get SIF embedding
        embedding_vector = SIF_embedding.SIF_embedding(self.We, x, w, params) # embedding[i,:] is the embedding for sentence i
        
        num_dims = embedding_vector.shape[1]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        return ExtractorResult(embedding_vector[0],
                               stim,
                               self,
                               features=features)
        

class AverageEmbeddingExtractor(DirectSentenceExtractor):
    
    _available_word_embeddings = ['glove','fasttext','word2vec']
    _fasttext  = 'fasttext'
    _fasttext_corpus = 'C'
    _word2vec = 'word2vec'
    _word2vec_corpus = 'google-news'


    def __init__(self,embedding='glove',dimensionality=300,\
               content_only=True,stopWords=None,corpus='840B',unk_vector=None,binary=False):
     
        print ('inside average embedding class')
        
        super(AverageEmbeddingExtractor, self).__init__()

        
        if not isinstance(embedding,str):
            raise ValueError('Type of embedding should be none or selected' + \
            ' from the list provided')
        
        if not isinstance(dimensionality,int):
            raise ValueError('Embedding dimension should be integer or' + \
            ' set None to use default')
        
        if not isinstance(content_only,bool):
            raise ValueError('Choice of using content word ' + \
             ' should be boolean (True/False). Default ' + \
            ' is True.')
        else:
            self.content_only=content_only
        
        if not isinstance(binary,bool):
            raise ValueError('Choice of nature of embedding; ' + \
             ' should be boolean (True/False). Default ' + \
            ' is False.')
        else:
            self.binary=binary
        
        if not isinstance(corpus,str):
            raise ValueError('Choice of corpus' + \
             ' should be string or unused')

        self.stop_words = nltk.corpus.stopwords.words('english')

        if not isinstance(stopWords,list):
            
            if stopWords !=None:
                raise ValueError('stopwords should be a list. Default' + \
                             ' is NLTK-stopwords. If you don\'t want to use + \
                                any stopwords use content_only = False instead')
        else:
            self.stop_words =  stopWords  
        
        if embedding.lower() not in self._available_word_embeddings:
            raise ValueError('selected embedding type is not ' + \
                             ' compatible with the embedding method, please check the available.' +\
                             ' embeddings')
        else:
            self.embedding = embedding
    
        ''' Check whether the embedding file exist '''
        if embedding == self._fasttext:
            '''for fasttext we only support the following 
               settings. Thus any other selection (i.e., 
               dimensionality < 300 or > 300) will be 
               superseded '''
            self.corpus = self._fasttext_corpus
            dimensionality = 300 
        
        if embedding == self._word2vec:
            '''for word2vec we only support the following 
               settings. Thus any other selection (i.e., 
               dimensionality < 300 or > 300) will be 
               superseded '''
            self.corpus = self._word2vec_corpus
            self.binary = True
            self._text = '.bin'
            dimensionality = 300 
        
        self.embedding_input =  self.embedding + self.corpus  + str(dimensionality) 
        self.embedding_path = os.path.join(self._embedding_model_path,self.embedding)
        self.embedding_file  = self.embedding_input + self._text
        self.embedding_file  = os.path.join(self.embedding_path,self.embedding_file)

        if not os.path.exists(self.embedding_file):
            raise ValueError('Embedding model file ' + self.embedding_file +  
                                ' is missing. Please download ' + \
                                'the model file(s) via running the following command: ' +\
                                'python download.py [embedding_name], where [embedding_name] ' +\
                                'is from [glove,word2vec,fasttext]. Also see README for ' + \
                                'the available models.')


    def _loadModel(self):
        self.wvModel = keyedvectors.KeyedVectors.load_word2vec_format(self.embedding_file,binary=self.binary,encoding='latin1')
        print(self.embedding + ': ' + 'model loaded')

    def _getModel(self):
        return self.wvModel 

    def _embed(self,stim):
        
        if not isinstance(stim,list):
            stim = [stim]
        
        ''' 
            We only implement average word embedding'
        '''
        '''we already have the embeddings loaded '''
        num_dims = self.wvModel.vector_size
        '''the stimuli is a list '''
        ''' we create average embedding. 
            Need to decide whether move the functionality
            to an util class later 
        '''
        for s in stim:
            complex_s = ComplexTextStim(text=s.lower())
            embeddings = self.transform(complex_s)
            embedding_average_vector = np.zeros(num_dims)
        
            numWords = 0
            for embedding in embeddings:
                if self.content_only == True:
                    if embedding.stim.data in self.stop_words or \
                        not embedding.stim.data.isalnum() :
                        continue
            
                for index,value in enumerate(embedding._data[0]):
                    embedding_average_vector[index] += value
            
                numWords+=1
        
            for index in range(num_dims):
                embedding_average_vector[index] /=  numWords

        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        return ExtractorResult(embedding_average_vector,
                               stim,
                               self,
                               features=features)
        

class Doc2vecExtractor(DirectSentenceExtractor):
    
    '''To read about doc2vec method please refer the paper - 
    Distributed Representations of Sentences and Documents 
    by Le and Mikolov'''
    
    
    def __init__(self):
        print ('inside Doc2vecExtractor class')
        
        super(Doc2vecExtractor, self).__init__()
        
        self._doc2vecEmbedding = 'doc2vec.bin'
        self._method = 'doc2vec'

        self._doc2vecEmbeddingFile = os.path.join(self._embedding_model_path,self._method,self._doc2vecEmbedding)

        if not os.path.exists(self._doc2vecEmbeddingFile):
            raise ValueError('Doc2vec model file ' + self._doc2vecEmbedding + \
                                ' is missing. Please download ' + \
                                ' the model files via the following command: ' +\
                                ' python download.py doc2vec' )
                    
        
    def _loadModel(self):
        
        self.doc2vecModel = doc2vecVectors.Doc2Vec.load(self._doc2vecEmbeddingFile)

    def _embed(self,stim):

        start_alpha=0.01
        infer_epoch=1000 
        
        if not isinstance(stim,list):
            stim = [stim]
        
        embedding_vectors = self.doc2vecModel.infer_vector(stim, alpha=start_alpha,
                                            steps=infer_epoch)
            
        num_dims = embedding_vectors.shape[0]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        
        return ExtractorResult(embedding_vectors,
                               stim,
                               self,
                           features=features)


class DANExtractor(DirectSentenceExtractor):
    
    '''Currently we have DAN (Deep Averaging Networks)
        encoding as a service running on Tensorflow Hub.
    '''
    _hub_path = "https://tfhub.dev/google/universal-sentence-encoder/2"
    _method = 'dan'

    
    def __init__(self):
        
        super(DANExtractor, self).__init__()
        self._hub_path = os.path.join(self._embedding_model_path, 'tfhub')
        os.environ["TFHUB_CACHE_DIR"] = self._hub_path
        self.dan_encoder = hub.Module(self._hub_path)
    
    def setOSEnv(self,path):
        '''to use TF hub we need to se the cache'''
        os.environ["TFHUB_CACHE_DIR"] = path
    
    def _embed(self,stim):
        
       # stim = [stim]
        if not isinstance(stim,list):
            stim = [stim]
       
       
        embeddings = self.dan_encoder(stim,as_dict=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embedding_vectors = sess.run(embeddings)    
        
        num_dims = (embedding_vectors.get('default')[0]).shape[0]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        
        embedding_dan_vectors = embedding_vectors.get('default')
        return ExtractorResult(embedding_dan_vectors,
                               stim,
                               self,
                               features=features)
        
class BertExtractor(DirectSentenceExtractor):
      
    '''For more information regarding Bert please
          refer to the original paper - 
          BERT: Pre-training of Deep Bidirectional Transformers for
        Language Understanding (Devlin et al.) and
        code  - https://github.com/google-research/bert
    ''' 
    
    '''we are providing an interface to the 
          extract_features.py script. For more details (i.e.,
          choice of Transformer layer) refer to the original paper
    '''
    ''' we are using mostly the default parameters for BERT 
    '''
    _batch_size = 8
    _layers = '-1,-2,-3,-4'
    _max_seq_length = 128
    _bert_config_file = 'bert_config.json'
    _do_lower_case = True
    _num_tpu_cores = 8
    _master = None
    _use_tpu = False
    _init_checkpoint = 'bert_model.ckpt'
    _use_one_hot_embeddings = False
    _method = 'bert'
    _model = 'bert_uncased_L-12_H-768_A-12'
    def __init__(self):
        
        super(BertExtractor, self).__init__()

        self._vocab_file = 'vocab.txt'
        
        self._bert_path = os.path.join(self._embedding_model_path,self._method,self._method,self._model)  
        
        if not os.path.exists(os.path.join(self._bert_path,self._vocab_file)):
            raise ValueError('BERT model file(s) '  + 
                                 ' '\
                                ' are missing. Please download ' + \
                                ' the model files via the following command: ' +\
                                ' python download.py bert ')

        
        self._layer_indexes = [int(x) for x in self._layers.split(",")]

        self._bert_config = bert_modeling.BertConfig.from_json_file(os.path.join(self._bert_path,self._bert_config_file))

        self._bert_tokenizer = bert_tokenization.FullTokenizer(
        vocab_file=os.path.join(self._bert_path,self._vocab_file), do_lower_case=self._do_lower_case)


    def _embed(self,stim):
        
        if not isinstance(stim,list):
            stim = [stim]
        
        embeddings = bert_extract_features.pliers_embedding(self._layer_indexes,\
                                                            self._bert_config,\
                                                            self._bert_tokenizer,\
                                                            stim,\
                                                            self._batch_size,\
                                                            self._num_tpu_cores,\
                                                            self._master,\
                                                            self._use_tpu,\
                                                            os.path.join(self._bert_path,self._init_checkpoint),\
                                                            self._use_one_hot_embeddings,\
                                                            self._max_seq_length)
                                                                                                             
        
        num_dims = embeddings[0].shape[0]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]

        return ExtractorResult(embeddings[0],
                               stim,
                               self,
                               features=features)

class ElmoExtractor(DirectSentenceExtractor):
    
    _options_file = 'elmo_2x2048_256_2048cnn_1xhighway_options.json'
    _weight_file = 'elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
    _method = 'elmo'
    
    def __init__(self,layer=None):
        
        super(ElmoExtractor, self).__init__()

        
        self._options_file = os.path.join(self._embedding_model_path,self._method,self._method,self._options_file)
        self._weight_file = os.path.join(self._embedding_model_path,self._method,self._method,self._weight_file)
        
        '''first check whether weight/option exist'''
        
        if not os.path.exists(self._options_file):
            raise ValueError('Elmo model file(s) '  + 
                                 ' '\
                                ' are missing. Please download ' + \
                                ' the model files via the following command: ' +\
                                ' python download.py(\'elmo\')' )

        
        
        self._elmoObj = ElmoEmbedder(options_file=self._options_file, \
                                 weight_file=self._weight_file)
        
        if layer == None:
            self._layer = 'default'
        else:
            self._layer = layer
        

        
    def _embed(self,stim):
        
        if not isinstance(stim,list):
            stim = [stim]
            
            
        embeddings = []
        for s in stim:
            tokens = s.split()
            embedding_layers = self._elmoObj.embed_sentence(tokens)
            if self._layer == 'default':
                '''we take the average from
                the three layers of Elmo'''
                embedding_avg = np.average(embedding_layers, axis=0)
                sentence_op = np.average(embedding_avg, axis=0)
            elif self._layer == 'top':
                embedding_top = embedding_layers[-1]
                sentence_op = np.average(embedding_top, axis=0)
            elif self._layer == 'bottom':
                embedding_bottom = embedding_layers[1]
                sentence_op = np.average(embedding_bottom, axis=0)
            
            embeddings.append(sentence_op)
        
        num_dims = embeddings[0].shape[0]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]

        return ExtractorResult(embeddings[0],
                               stim,
                               self,
                               features=features)

 
class ElmoTFHubExtractor(DirectSentenceExtractor):
    
    '''Currently we have Elmo encoding as a service
        running on Tensorflow Hub. '''
    
    '''For more information please refer to the original 
        paper - Deep contextualized word representations 
        (Peters et al.). Using TF hub it is feasible
        to extract the default as well as output from each
        of Elmos's LSTM networks. We also provide a 
        mean-pool output from the LSTM networks'''
    
    os.environ["TFHUB_CACHE_DIR"] = '/Users/mit-gablab/work/data_workspace/tfhub/'
    _hub_path = "https://tfhub.dev/google/elmo/2"
    _method = 'elmo'
    _lstm1 = 'lstm1'
    _lstm2 = 'lstm2'
    _both = 'both'
    _default = 'default'
    
    
    def __init__(self,layer=None):

        self.elmo_encoder = hub.Module(self._hub_path,trainable=True)
        if layer == None:
            self._layer = 'default'
        else:
            self._layer = layer

        super(ElmoTFHubExtractor, self).__init__()
        
        
    def _embed(self,stim):
        
        if not isinstance(stim,list):
            stim = [stim]
        
        embeddings = self.elmo_encoder(stim,as_dict=True,signature="default")
        
        default_elmo = embeddings["default"]
        lstm_outputs1 = embeddings["lstm_outputs1"]
        lstm_outputs2 = embeddings["lstm_outputs2"]
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            
            if self._layer == self._default:
                embedding_vectors_1 = sess.run(default_elmo)

            elif self._layer == self._lstm1:
                embedding_vectors_1 = sess.run(lstm_outputs1)
                
            elif self._layer == self._lstm2:
                embedding_vectors_1 = sess.run(lstm_outputs2)
                
            elif self._layer == self._both:
                embedding_vectors_1 = sess.run(lstm_outputs1)
                embedding_vectors_2 = sess.run(lstm_outputs2)


        num_dims = embedding_vectors_1.shape[0]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]

        if self._layer == self._default: 
            return ExtractorResult(embedding_vectors_1,
                               stim,
                               self,
                               features=features)
              

        embedding_elmo_vectors = []
        for index in range(0,len(stim)):
            embedding_elmo_vector = embedding_vectors_1[index]
            if self._layer == self._lstm1 or self._layer == self._lstm2: 
                embedding_elmo_vector = np.mean(embedding_elmo_vector,axis=0)
            if self._layer == self._both : 
                embedding_elmo_vector_2 = embedding_vectors_2[index]

                embedding_elmo_vector = np.concatenate(np.mean(embedding_elmo_vector,axis=0),\
                                        np.mean(embedding_elmo_vector_2,axis=0))

            embedding_elmo_vectors.append(embedding_elmo_vector)
            
        return ExtractorResult(embedding_elmo_vectors,
                               stim,
                               self,
                               features=features)
        

class SkipThoughtExtractor(DirectSentenceExtractor):
    
    '''To learn about skipthought extractor please
        refer Skip-Thought Vectors paper by Kiros et al.'''
    
    '''Note: skipthought is based on Theano 1.0.4 and
        Numpy <1.16.3 (we are using numpy-1.16.2)'''
    
    _skipthought_files = ['dictionary.txt','utable.npy','btable.npy',\
                          'uni_skip.npz','uni_skip.npz.pkl','bi_skip.npz',\
                          'bi_skip.npz.pkl']
    
    _method = 'skipthought'
    
    def __init__(self):
        print ('inside skipthought class')
        super(SkipThoughtExtractor, self).__init__()

        
        for _skipthought_file in self._skipthought_files:
            if not os.path.exists(self._embedding_model_path+self._method+'/'+_skipthought_file):
                raise ValueError('Skipthought model file ' + _skipthought_file + 
                                 ' '\
                                ' is missing. Please download ' + \
                                ' the model files via the following command: ' +\
                                ' python download.py skipthought' )
                    
        
        
    def _loadModel(self):

        self.skipthought_model = skipthoughts.load_model()
        self.skipthought_encoder = skipthoughts.Encoder(self.skipthought_model)

    def _embed(self,stim):
        
        if not isinstance(stim,list):
            stim = [stim]

        embedding_vectors = self.skipthought_encoder.encode(stim,verbose=False)
        
        num_dims = embedding_vectors.shape[1]
        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        return ExtractorResult(embedding_vectors[0],
                               stim,
                               self,
                               features=features)
