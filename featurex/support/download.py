import nltk

CORPORA = ['punkt', 'maxent_treebank_pos_tagger', 'averaged_perceptron_tagger']


def download_nltk_data():
    ''' Download nltk corpora required for one or more feature extractors. '''
    for c in CORPORA:
        nltk.download(c)

if __name__ == '__main__':
    download_nltk_data()
