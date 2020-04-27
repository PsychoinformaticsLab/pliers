''' The `Converter` hierarchy contains Transformer classes that take an object
of arbitrary class (but almost always a `Stim` subclass) as input, and return a
`Stim` instance (of different class) as output.
'''

from .api import (WitTranscriptionConverter,
                  IBMSpeechAPIConverter,
                  GoogleSpeechAPIConverter,
                  GoogleVisionAPITextConverter,
                  MicrosoftAPITextConverter,
                  RevAISpeechAPIConverter)
from .base import Converter, get_converter
from .image import TesseractConverter
from .iterators import (VideoFrameIterator, VideoFrameCollectionIterator,
                        ComplexTextIterator)
from .multistep import VideoToTextConverter, VideoToComplexTextConverter
from .video import VideoToAudioConverter
from .misc import ExtractorResultToDFConverter


__all__ = [
    'WitTranscriptionConverter',
    'GoogleSpeechAPIConverter',
    'IBMSpeechAPIConverter',
    'GoogleVisionAPITextConverter',
    'TesseractConverter',
    'VideoFrameIterator',
    'VideoFrameCollectionIterator',
    'ComplexTextIterator',
    'MicrosoftAPITextConverter',
    'VideoToTextConverter',
    'VideoToComplexTextConverter',
    'VideoToAudioConverter',
    'RevAISpeechAPIConverter',
    'ExtractorResultToDFConverter',
    'Converter',
    'get_converter'
]
