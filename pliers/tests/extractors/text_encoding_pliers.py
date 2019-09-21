import warnings
import sys
import argparse
import numpy as np 
import logging


from pliers.extractors.text_encoding import DirectSentenceExtractor,\
embedding_methods,DirectTextExtractorInterface


logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(name='text_encoding_logger')


def textExtractor(ext,method,inputFile,num=None,fileType=None,\
                  embedding_type=None,cbow=False):
    
    f = open(inputFile)
    '''id - text (tab separated)'''
    allInputs = [line.strip() for line in f]
    
    #allInputs = allInputs[0:5]
    logger.info('length of input: ' + str(len(allInputs)))
    
    allResults = []
    allStimulis = []
    
    for input in allInputs:
        results = ext.embed(input.lower(),cbow=cbow)
        allResults.extend(results._data)
        allStimulis.extend(results.stim)
        
    logger.info('length results ' + str(len(allResults)))
    logger.info('length stimuli ' + str(len(allStimulis)))
    
    
    return allResults

def parseArguments():
    
    parser = argparse.ArgumentParser(description='Text encoding via Pliers')
    parser.add_argument('--method_name', type=str, default='averageWordEmbedding',
                         help = 'text encoding via word or sentence embedding. Note, ' + 
                         'if the selected method is not average embedding, ' + 
                         'e.g., Elmo or Bert, subsequentt arguments will be ' + 
                         'neglected.')
    parser.add_argument('--embedding', type=str, default='glove',
                         help = 'select from glove, word2vec, or fasttext')
    parser.add_argument('--dimensionality', type=int, default=300, 
                        help = 'length of embedding vectors')
    parser.add_argument('--corpus', type=str, default='42B', 
                        help = 'corpus used to create embedding')
    parser.add_argument('--content_only', type=bool, default=True,
                        help = 'whether only content words are used')
    parser.add_argument('--stopWords', type=list, default=None, 
                        help = 'list of stopwords, unless provided ' + 
                        'stopwords from NLTK is used')
    parser.add_argument('--unk_vector', type=list, default=None,
                        help = 'particular vector for unknown words (default is zero) ')
    parser.add_argument('--binary', type=bool, default=False)

    parser.add_argument('--input',type=str,required=True, 
                        help = 'input file (required) ')
    parser.add_argument('--output',type=str,required=True,
                        help = 'output file (by default the ' + 
                        'output is numpy output') 
    
    parser.add_argument('--cbow',type=bool,default=False,
                        help = 'whether the user expects embeddings for each ' +
                         'cbow (default is false ') 
    
    
    args = parser.parse_args()
    logger.info(args)
    
    return args
    
    
def main():
    
    arguments = parseArguments()
    
    method = arguments.method_name
    embedding = arguments.embedding
    dim = arguments.dimensionality
    corpus = arguments.corpus
    content_only = arguments.content_only
    stopWords = arguments.stopWords
    unk_vector = arguments.unk_vector
    binary = arguments.binary
    inputFile = arguments.input
    outputFile = arguments.output
    cbow = arguments.cbow
    
    
    extractor = DirectTextExtractorInterface(method=method,\
                                             embedding=embedding,\
                                             dimensionality=dim,\
                                             corpus=corpus,\
                                             content_only=content_only,\
                                             binary = binary,\
                                             stopWords=stopWords,\
                                             unk_vector=unk_vector)
    vectors = textExtractor(extractor,method=method,\
                            inputFile=inputFile,cbow=cbow)
    np.savetxt(outputFile, vectors)
    
    logger.info('output encodings of size %s.' % len(vectors))
    
if __name__ == '__main__':
    
    main()
    
    

