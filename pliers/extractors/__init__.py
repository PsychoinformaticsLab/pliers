from .base import Extractor, ExtractorResult, merge_results
from .api import IndicoAPIExtractor, ClarifaiAPIExtractor
from .audio import STFTAudioExtractor, MeanAmplitudeExtractor
from .google import (GoogleVisionAPIFaceExtractor,
                     GoogleVisionAPILabelExtractor,
                     GoogleVisionAPIPropertyExtractor)
from .image import (BrightnessExtractor, SaliencyExtractor, SharpnessExtractor,
                    VibranceExtractor)
from .text import (ComplexTextExtractor, DictionaryExtractor,
                   PredefinedDictionaryExtractor, LengthExtractor,
                   NumUniqueWordsExtractor, PartOfSpeechExtractor)
from .video import (DenseOpticalFlowExtractor)

# __all__ = ['api', 'audio', 'google', 'image', 'text', 'video']
__all__ = [
    'ExtractorResult',
    'IndicoAPIExtractor',
    'ClarifaiAPIExtractor',
    'STFTAudioExtractor',
    'MeanAmplitudeExtractor',
    'GoogleVisionAPIFaceExtractor',
    'GoogleVisionAPILabelExtractor',
    'GoogleVisionAPIPropertyExtractor',
    'BrightnessExtractor',
    'SaliencyExtractor',
    'SharpnessExtractor',
    'VibranceExtractor'
    'ComplexTextExtractor',
    'DictionaryExtractor',
    'PredefinedDictionaryExtractor',
    'LengthExtractor',
    'NumUniqueWordsExtractor',
    'PartOfSpeechExtractor',
    'DenseOpticalFlowExtractor',
    'merge_results'
]

