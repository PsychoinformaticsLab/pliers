''' Support utilities for (mostly) text data. '''

# TODO: This stuff should probably be merged into .utils.

from .decorators import requires_nltk_corpus
from .download import download_nltk_data

__all__ = [
    'requires_nltk_corpus',
    'download_nltk_data'
]
