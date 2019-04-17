import warnings
import gensim
import sklearn.metrics.pairwise
import numpy as np 
from nltk import __maintainer__

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from ...pliers.stimuli import TextStim, ComplexTextStim
from pliers.extractors import WordEmbeddingExtractor
from pliers.extractors.text_encoding import DirectSentenceExtractor,\
embedding_methods,DirectTextExtractorInterface
from pliers import config
config.set_option('cache_transformers', False)




import tensorflow as tf
import tensorflow_hub as hub

'''
import hashlib
#handle = "https://tfhub.dev/google/universal-sentence-encoder/2"
handle = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
print(hashlib.sha1(handle.encode("utf8")).hexdigest())
'''



def collectContineousInput (inputPath,sequence,num):
    
    inputFile  = 'stim_'+num+'sentences_dereferencedpronouns_contineous.txt'

    f = open(inputPath+inputFile)
    contineousLines = f.readlines()
    
    inputFile  = 'stim_'+num+'sentences_dereferencedpronouns.txt'

    f = open(inputPath+inputFile)
    sequenceLines = f.readlines()
    
    if sequence == True:
        return [contineousLines,sequenceLines]
    else:
        return [contineousLines]

def textExtractorBertService(method,num,fileType,embedding_type=None):
    
    inputPath = '/Users/mit-gablab/work/brain_decoding/data/input_text/'

    if fileType == 'sep':
        fileName = inputPath + '/new/' + 'stim_'+str(num)+'sentences_dereferencedpronouns.txt'
        
        if embedding_type is None:
            embeddingFile  = num+'separate.'+method+'.npy'
        else:
            embeddingFile  = num+'separate.'+method+'.'+embedding_type+'.npy'
            
        
    elif fileType == 'all':
        fileName = inputPath + '/new/' +str(num)+'contineous.all.txt'
        if embedding_type is None:
            embeddingFile = num+'contineous.all.'+method+'.npy'
        else:
            embeddingFile = num+'contineous.all.'+method+'.'+embedding_type+'.npy'

    
    elif fileType == 'last':
        fileName = inputPath + '/new/' +str(num)+'contineous.last.txt'
        if embedding_type is None:
            embeddingFile = num+'contineous.last.'+method+'.npy'
        else:
            embeddingFile = num+'contineous.last.'+method+'.'+embedding_type+'.npy'

    
    f = open(fileName)
    allInputs = [line.strip().split('\t')[1] for line in f]
    
    print('length of input: ' + str(len(allInputs)))
    print (allInputs[0:5])
    
    allResults = []
    
    import bert_serving
    import numpy as np
    from bert_serving.client import BertClient
    bc = BertClient()
    
    allResults = bc.encode(allInputs)
    np.save(inputPath+'/new/'+embeddingFile, allResults)
    
    '''sanity checking for data length'''
    textRepresentations = np.load(inputPath+'/new/'+embeddingFile)
    print("Loaded encodings of size %s.", textRepresentations.shape)

  

def textExtractor(ext,method,num,fileType,embedding_type=None):
    
    inputPath = '/Users/mit-gablab/work/brain_decoding/data/input_text/'

    if fileType == 'sep':
        fileName = inputPath + '/new/' + 'stim_'+str(num)+'sentences_dereferencedpronouns.txt'
        
        if embedding_type is None:
            embeddingFile  = num+'separate.'+method+'.npy'
        else:
            embeddingFile  = num+'separate.'+method+'.'+embedding_type+'.npy'
            
        
    elif fileType == 'all':
        fileName = inputPath + '/new/' +str(num)+'contineous.all.txt'
        if embedding_type is None:
            embeddingFile = num+'contineous.all.'+method+'.npy'
        else:
            embeddingFile = num+'contineous.all.'+method+'.'+embedding_type+'.npy'

    
    elif fileType == 'last':
        fileName = inputPath + '/new/' +str(num)+'contineous.last.txt'
        if embedding_type is None:
            embeddingFile = num+'contineous.last.'+method+'.npy'
        else:
            embeddingFile = num+'contineous.last.'+method+'.'+embedding_type+'.npy'

    
    f = open(fileName)
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
            
    np.save(inputPath+'/new/'+embeddingFile, allResults)
    
    '''sanity checking for data length'''
    textRepresentations = np.load(inputPath+'/new/'+embeddingFile)
    print("Loaded encodings of size %s.", textRepresentations.shape)
    


def main():
    
    nums = ['384','243','256']
    reps = ['averagewordembedding','doc2vec','infersent','skipthought',\
               'elmo','dan','sif'] 
    
    '''note, skipthought/infersent is not made for > 1 sentence 
    so we can take average
    '''
    reps = ['elmo']
    
    type = ['sep','all','last']
    
    for rep in reps:
        ext = DirectTextExtractorInterface(method=rep)
        for num in nums:  
            for t in type:  
                textExtractor(ext,rep,num,t)
    

if __name__ == '__main__':
    
    main()
    

