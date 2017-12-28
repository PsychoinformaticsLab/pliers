''' Support utilities. '''

from .decorators import requires_nltk_corpus
from .download import download_nltk_data

__all__ = [
    'requires_nltk_corpus',
    'download_nltk_data'
]
