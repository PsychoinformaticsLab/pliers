''' The `Filter` hierarchy contains Transformer classes that take a `Stim`
of one type as input and return a `Stim` of the same type as output (but with
some changes to its data).
'''

from .audio import (AudioTrimmingFilter, 
                    AudioResamplingFilter)
from .base import TemporalTrimmingFilter
from .image import (ImageCroppingFilter,
                    ImageResizingFilter,
                    PillowImageFilter)
from .text import (WordStemmingFilter,
                   TokenizingFilter,
                   TokenRemovalFilter,
                   PunctuationRemovalFilter,
                   LowerCasingFilter)
from .video import (FrameSamplingFilter,
                    VideoTrimmingFilter)


__all__ = [
    'AudioTrimmingFilter',
    'AudioResamplingFilter',
    'TemporalTrimmingFilter',
    'ImageCroppingFilter',
    'ImageResizingFilter',
    'PillowImageFilter',
    'WordStemmingFilter',
    'TokenizingFilter',
    'TokenRemovalFilter',
    'PunctuationRemovalFilter',
    'LowerCasingFilter',
    'FrameSamplingFilter',
    'VideoTrimmingFilter'
]
