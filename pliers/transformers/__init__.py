''' The `transformers` module contains the base `Transformer` class from which
all other pliers transformers inherit, as well as `Transformer` subclasses
that have multiple subclasses spanning different modules (e.g., Google Cloud
extractors that span audio, image, etc.).'''

from .base import Transformer, BatchTransformerMixin, get_transformer
from .api import (GoogleAPITransformer,
                  GoogleVisionAPITransformer,
                  MicrosoftAPITransformer,
                  MicrosoftVisionAPITransformer)


__all__ = [
    'BatchTransformerMixin',
    'get_transformer',
    'GoogleAPITransformer',
    'GoogleVisionAPITransformer',
    'MicrosoftAPITransformer',
    'MicrosoftVisionAPITransformer',
    'Transformer'
]
