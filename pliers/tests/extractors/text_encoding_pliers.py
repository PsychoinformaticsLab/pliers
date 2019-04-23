import warnings
import gensim
import sklearn.metrics.pairwise
import numpy as np 
from nltk import __maintainer__
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import OutputProjectionWrapper

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from pliers.stimuli import TextStim, ComplexTextStim
from pliers.extractors import WordEmbeddingExtractor
from pliers.extractors.text_encoding import DirectSentenceExtractor,\
embedding_methods,DirectTextExtractorInterface
from pliers import config
config.set_option('cache_transformers', False)

import sys


import tensorflow as tf
import tensorflow_hub as hub

def textExtractor(ext,method,inputFile,num=None,fileType=None,embedding_type=None):
    
    f = open(inputFile)
    '''id - text (tab separated)'''
    allInputs = [line.strip().split('\t')[1] for line in f]
    
    print('length of input: ' + str(len(allInputs)))
    print (allInputs[0:5])
    
    allResults = []
    
    if method == 'dan' or method == 'elmo':
        allResults.extend(ext._embed(allInputs))
    else:
        for input in allInputs:
            results = ext._embed(input.lower())
            allResults.append(results)
            
    return allResults

def main(args):
    
    reps = ['averagewordembedding','doc2vec','infersent','skipthought',\
               'elmo','dan','sif'] 
    
    '''note, skipthought/infersent is not made for > 1 sentence 
    so we can take average
    '''
    OutputPath = args[0]
    inputFile = args[1]
    
    for rep in reps:
        extractor = DirectTextExtractorInterface(method=rep)
        vectors = textExtractor(extractor,rep,inputFile)
    
        embeddingFile = 'embedding_' + rep + '.npy'
        np.save(OutputPath+embeddingFile, vectors)
        '''sanity checking for data length'''
        textRepresentations = np.load(OutputPath+embeddingFile)
        print("Loaded encodings of size %s.", textRepresentations.shape)
    
if __name__ == '__main__':
    
    main(sys.argv[1:])
    

