''' The `Extractor` hierarchy contains Transformer classes that take a `Stim`
of any type as input and return extracted feature information (rather than
another `Stim` instance).
'''

from .base import Extractor, ExtractorResult, merge_results
from .api import (IndicoAPITextExtractor,
                  IndicoAPIImageExtractor,
                  ClarifaiAPIExtractor)
from .audio import (STFTAudioExtractor,
                    MeanAmplitudeExtractor,
                    SpectralCentroidExtractor,
                    SpectralBandwidthExtractor,
                    SpectralContrastExtractor,
                    SpectralRolloffExtractor,
                    PolyFeaturesExtractor,
                    RMSEExtractor,
                    ZeroCrossingRateExtractor,
                    ChromaSTFTExtractor,
                    ChromaCQTExtractor,
                    ChromaCENSExtractor,
                    MelspectrogramExtractor,
                    MFCCExtractor,
                    TonnetzExtractor,
                    TempogramExtractor)
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
                   NumUniqueWordsExtractor, PartOfSpeechExtractor,
                   WordEmbeddingExtractor, TextVectorizerExtractor,
                   VADERSentimentExtractor)
from .video import (FarnebackOpticalFlowExtractor)

__all__ = [
    'Extractor',
    'ExtractorResult',
    'IndicoAPITextExtractor',
    'IndicoAPIImageExtractor',
    'ClarifaiAPIExtractor',
    'STFTAudioExtractor',
    'MeanAmplitudeExtractor',
    'SpectralCentroidExtractor',
    'SpectralBandwidthExtractor',
    'SpectralContrastExtractor',
    'SpectralRolloffExtractor',
    'PolyFeaturesExtractor',
    'RMSEExtractor',
    'ZeroCrossingRateExtractor',
    'ChromaSTFTExtractor',
    'ChromaCQTExtractor',
    'ChromaCENSExtractor',
    'MelspectrogramExtractor',
    'MFCCExtractor',
    'TonnetzExtractor',
    'TempogramExtractor',
    'GoogleVisionAPIFaceExtractor',
    'GoogleVisionAPILabelExtractor',
    'GoogleVisionAPIPropertyExtractor',
    'GoogleVisionAPISafeSearchExtractor',
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
    'FarnebackOpticalFlowExtractor',
    'WordEmbeddingExtractor',
    'TextVectorizerExtractor',
    'VADERSentimentExtractor',
    'merge_results'
]
