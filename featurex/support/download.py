import nltk

CORPORA = ['punkt']


def download_nltk_data():
    ''' Download nltk corpora required for one or more feature extractors. '''
    for c in CORPORA:
        nltk.download(c)

if __name__ == '__main__':
    download_nltk_data()
