import nltk
import os
import requests
import sys
import tarfile
import logging

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec,get_glove_info
import argparse



CORPORA = [
    'punkt',
    'maxent_treebank_pos_tagger',
    'averaged_perceptron_tagger',
    'vader_lexicon'
]



_available_word_embeddings = ['glove','fasttext','word2vec']
_aws_bucket_path = 'https://s3.amazonaws.com/mlt-word-embeddings/'
_tar_gz_extn = '.tar.gz'

_current_path = os.path.abspath(os.path.dirname(__file__))
_embedding_model_path = os.path.join(_current_path,'../../datasets/embeddings/')


def download_nltk_data():
    ''' Download nltk corpora required for one or more feature extractors. '''
    for c in CORPORA:
        nltk.download(c)

def prepare_keyvector(embedding,_embedding_model_path):
    
    allFiles = os.listdir(_embedding_model_path)
    vectorFiles = []
    
    for file in allFiles:
        if 'vectors' in file:
            vectorFiles.append(file)

    for vectorFile in vectorFiles:

        embed_type = vectorFile.split('_')[0]
        vocabFile = embed_type + '_vocabulary.txt'
        if embed_type.startswith('GV'):
            embed_type = embed_type.replace('GV','glove')
        elif embed_type.startswith('W2V'):
            embed_type = embed_type.replace('W2V','word2vec')
        elif embed_type.startswith('FT'):
            embed_type = embed_type.replace('FT','fasttext')
        else:
            print('embedding type is not supported, check.')

        combine(os.path.join(_embedding_model_path,embed_type),os.path.join(_embedding_model_path,vocabFile),
                os.path.join(_embedding_model_path,vectorFile))
        print('done: ' + vectorFile)

def combine(combineFile,vocabFile,vectorFile):
    
    numLines, numDims = get_glove_info(vectorFile)
    
    vectors = []
    vocabs = []
    with open(vectorFile, 'r',encoding='latin1') as fin:
            for line in fin:
                vectors.append(line.strip())

    with open(vocabFile, 'r',encoding='latin1') as fin:
            for line in fin:
                vocabs.append(line.strip())
                
    with gensim.utils.smart_open(combineFile+'.txt', 'w',encoding='latin1') as fout:
        fout.write("{0} {1}\n".format(numLines, numDims+1))
        for index,vector in enumerate(vectors):
            vocab = vocabs[index]
        #    fout.write(vocab + ' '+vector)
            fout.write("{0} {1}\n".format(vocab, vector))

    
            
def download_embedding_data(embedding='glove',keep_download=False):
    
    embedding_aws_file = os.path.join(_aws_bucket_path,embedding)
    embedding_aws_file = embedding_aws_file + _tar_gz_extn
    embedding_model_file= embedding_aws_file.split('/')[-1]
    embedding_model_folder = embedding_model_file.split('.')[0]
    embedding_model_full_path = os.path.join(_embedding_model_path,embedding_model_folder)
    embedding_local_file = os.path.join(_embedding_model_path, embedding_model_file)
    if not os.path.exists(embedding_local_file):
        _download_pretrained_embedding_model(embedding_model_full_path,\
                                            embedding_model_file,\
                                             embedding_local_file,\
                                             embedding_aws_file)
    
        
    else:
        print('Embedding zip file already exist, please check whether it has been unzipped correctly.')
        logging.info('Embedding tar file already exist, please check whether it has been unzipped correctly.')
    
    _pretrained_lms = ['skipthought', 'bert','elmo',
                       'sif','doc2vec']
    '''currently the vectors and vocabulary are in
        different files and we need to combine them.
        This would be replaced in the next release'''
    if embedding not in _pretrained_lms:
        prepare_keyvector(embedding,_embedding_model_path)
        
    '''Remove the zip file since it might take up space'''
    if keep_download == False:
        if os.path.exists(embedding_local_file):
            os.remove(embedding_local_file)

#    return embedding_model_full_path

def _download_pretrained_embedding_model(embedding_model_full_path,\
                                         embedding_model_file,\
                                             embedding_local_file,\
                                             embedding_aws_file):
        
    print('Downloading Embedding model: ' +embedding_model_file)
    if not os.path.exists(embedding_model_full_path):
        os.makedirs(embedding_model_full_path)
        
    try:
        if not os.path.exists(embedding_local_file):
            r = requests.get(embedding_aws_file,stream=True)
                
            with open(embedding_local_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024): 
                    if chunk:
                        f.write(chunk)
    except IOError as e:
            #catch exception
        print(e)
            
    try:
        size = os.stat(embedding_local_file).st_size
        print('\tSuccessfully downloaded', embedding_model_file, size, 'bytes.')
        logging.info('\tSuccessfully downloaded', embedding_model_file, size, 'bytes.')
        tarfile.open(embedding_local_file, 'r:gz').extractall(embedding_model_full_path)
        
    except IOError as e:
        print(e)


def parseArguments():
    
    parser = argparse.ArgumentParser(description='Download required model/embedding files for Text encoding via Pliers')
    parser.add_argument('--pretrained', type=str, default='glove',
                         help = 'select from glove, word2vec, fasttext, bert etc. (check the documentation ' + 
                         'for the various options. ')
    parser.add_argument('--keep_download', type=bool, default=False, required=False, 
                        help = 'whether storing the zip/tar file or removing after the successfull download action; ' +
                            ' default is removing the zip file.' )

    args = parser.parse_args()
    print(args)
    
    return args

def main():
    
    arguments = parseArguments()
    pretrained = arguments.pretrained
    keep_download = arguments.keep_download


    '''downloading particular embedding file from 
        AWS'''
    '''We support three embeddings (e.g., glove,
        fasttext, word2vec) and several other
        pretrained models and LMs (e.g., skipthought, BERT, 
        ELMO).'''
    
    download_embedding_data(embedding=pretrained,keep_download=keep_download)

if __name__ == '__main__':
   main()
   
