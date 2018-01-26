''' Filters that operate on TextStim inputs. '''

from pliers.stimuli import AudioStim
from .base import Filter, TemporalTrimmingFilter


class AudioFilter(Filter):

    ''' Base class for all VideoFilters. '''

    _input_type = AudioStim


class AudioTrimmingFilter(TemporalTrimmingFilter, AudioFilter):
    pass
