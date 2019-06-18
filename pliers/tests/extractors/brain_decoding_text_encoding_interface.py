import warnings
import gensim
import sklearn.metrics.pairwise
import numpy as np 
from nltk import __maintainer__

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from pliers.stimuli import TextStim, ComplexTextStim
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


def getTopics(concepts,index=None):
    
    allTopics = []
    if index == None:
        '''need to return all topics'''
        for topics in concepts:
            if topics == 'BLANK':
                topics = ' '
            allTopics.append(topics)
    else:
        topics = concepts[index]
        if topics == 'BLANK':
            topics = ' '
        allTopics.append(topics)

    return allTopics
  

def textExtractor(ext,method,num,fileType,\
                  concepts=None,embedding_type=None):
    
    inputPath = '/Users/mit-gablab/work/brain_decoding/data/input_text/'

    if fileType == 'concept':
        
        fileName = inputPath + '/new/' + 'stim_'+str(num)+'sentences_dereferencedpronouns.txt'
        if embedding_type is None:
            embeddingFile  = num+fileType+'.'+method+'.npy'
        else:
            embeddingFile  = num+fileType+'.'+method+'.'+embedding_type+'.npy'

    elif fileType == 'repeat':
        
        fileName = inputPath + '/new/' + 'stim_'+str(num)+'sentences_dereferencedpronouns.txt'
        if embedding_type is None:
            embeddingFile  = num+fileType+'.'+method+'.npy'
        else:
            embeddingFile  = num+fileType+'.'+method+'.'+embedding_type+'.npy'
       

    elif fileType == 'sep':
        fileName = inputPath + '/new/' + 'stim_'+str(num)+'sentences_dereferencedpronouns.txt'
        
        if embedding_type is None:
            embeddingFile  = num+'separate.'+method+'temp.npy'
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
    data_type = 'not-mixed'
    if data_type == 'mixed':
        allInputs = []
        for line in f:
            elements = line.strip().split('\t')
            if len(elements) == 1:
                allInputs.append(elements[0])
            else:
                allInputs.append(elements[1])
    else:
        allInputs = [line.strip().split('\t')[1] for line in f]
    
    print('length of input: ' + str(len(allInputs)))
    print (allInputs[0:5])
    
    allResults = []
    
    if method == 'dan' or method == 'elmo':
        
        if fileType == 'concept' or fileType == 'repeat':
            topics = getTopics(concepts,index=None)
            allInputsTopics = []
            for index,input in enumerate(allInputs):
                input = input + ' ' + topics[index]
                allInputsTopics.append(input)
            allResults.extend(ext._embed(allInputsTopics)._data)
        else:
            allResults.extend(ext._embed(allInputs)._data)
    else:
        for index,input in enumerate(allInputs):
            if fileType == 'concept' or fileType == 'repeat':
                
                topics = getTopics(concepts,index=index)
                input = input + ' ' + topics[0] #its a list
                
            results = ext._embed(input.lower())
                
                #results_concept = ext._embed(concepts[index])
            '''
                for index,value in enumerate(results):
                        concept_value = results_concept[index]
                        results[index] += concept_value
            '''
            allResults.append(results._data)
    
    np.save(inputPath+'/new/'+embeddingFile, allResults)
    
    #sanity checking for data length
    textRepresentations = np.load(inputPath+'/new/'+embeddingFile)
    print("Loaded encodings of size %s.", textRepresentations.shape)
    
def getRepeats(num):
    
    inputPath = '/Users/mit-gablab/work/brain_decoding/data/input_text/'
    fileName = inputPath + '/new/' + str(num)+'repeat.txt'
    f =open(fileName)
    concepts = [line.strip()
                 for line in f ]
    return concepts



def getConcepts(num,top_concepts):
    
    inputPath = '/Users/mit-gablab/work/brain_decoding/data/input_text/'
    fileName = inputPath + '/new/' + str(num)+'topgloveconcepts.txt'
    f =open(fileName)
    concepts = [' '.join(line.strip().split('\t')[1].split(',')[0:top_concepts]) \
                 for line in f ]
    return concepts


def main():
    
    nums = ['384','243','256']
#    nums = ['180']
#    nums = ['158']
    reps = ['averagewordembedding','doc2vec','infersent','skipthought',\
               'elmo','dan','sif','fasttext'] 
    reps = ['averagewordembedding']
    reps = ['doc2vec']
    reps = ['skipthought']
    
    '''note, skipthought/infersent is not made for > 1 sentence 
    so we can take average
    '''
 #   reps = ['averagewordembedding']
    
    type = ['sep','all','last']
#    type = ['sep'],3
#    type = ['repeat'] # adding separate with concept
#    type = ['sep']
    embeddings = ["glove","fasttext","word2vec"]
    top_concepts =3
    concepts=None
    for embedding in embeddings:
        
        for rep in reps:
            ext = DirectTextExtractorInterface(method=rep,embedding=embedding,content_only=True)
            for num in nums:  
                for t in type:  
                    if t == 'concept':
                        concepts = getConcepts(num,top_concepts)
                    elif t == 'repeat':
                        concepts = getRepeats(num)
                            
                    textExtractor(ext,rep,num,t,concepts=concepts,embedding_type=embedding)
    

if __name__ == '__main__':
    
    main()
    

