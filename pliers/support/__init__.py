''' Support utilities for (mostly) text data. '''

# TODO: This stuff should probably be merged into .utils.

from .decorators import requires_nltk_corpus
from .download import download_nltk_data
from .setup_yamnet import setup_yamnet

__all__ = [
    'requires_nltk_corpus',
    'download_nltk_data',
    'setup_yamnet'
]
