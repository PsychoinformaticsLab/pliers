''' Extractor hierarchy. '''

from .base import Extractor, ExtractorResult, merge_results
from .api import (IndicoAPITextExtractor,
                  IndicoAPIImageExtractor,
                  ClarifaiAPIExtractor)
from .audio import (STFTAudioExtractor,
                    MeanAmplitudeExtractor,
                    RMSEExtractor)
from .google import (GoogleVisionAPIFaceExtractor,
                     GoogleVisionAPILabelExtractor,
                     GoogleVisionAPIPropertyExtractor,
                     GoogleVisionAPISafeSearchExtractor,
                     GoogleVisionAPIWebEntitiesExtractor)
from .image import (BrightnessExtractor, SaliencyExtractor, SharpnessExtractor,
                    VibranceExtractor)
from .models import TensorFlowInceptionV3Extractor
from .text import (ComplexTextExtractor, DictionaryExtractor,
                   PredefinedDictionaryExtractor, LengthExtractor,
                   NumUniqueWordsExtractor, PartOfSpeechExtractor)
from .video import (DenseOpticalFlowExtractor)

__all__ = [
    'Extractor',
    'ExtractorResult',
    'IndicoAPITextExtractor',
    'IndicoAPIImageExtractor',
    'ClarifaiAPIExtractor',
    'STFTAudioExtractor',
    'MeanAmplitudeExtractor',
    'RMSEExtractor',
    'GoogleVisionAPIFaceExtractor',
    'GoogleVisionAPILabelExtractor',
    'GoogleVisionAPIPropertyExtractor',
    'GoogleVisionAPIWebEntitiesExtractor',
    'BrightnessExtractor',
    'SaliencyExtractor',
    'SharpnessExtractor',
    'VibranceExtractor',
    'TensorFlowInceptionV3Extractor',
    'ComplexTextExtractor',
    'DictionaryExtractor',
    'PredefinedDictionaryExtractor',
    'LengthExtractor',
    'NumUniqueWordsExtractor',
    'PartOfSpeechExtractor',
    'DenseOpticalFlowExtractor',
    'merge_results'
]
