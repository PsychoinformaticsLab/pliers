''' Filters that operate on TextStim inputs. '''

from pliers.stimuli import AudioStim
from .base import Filter


class AudioFilter(Filter):

    ''' Base class for all VideoFilters. '''

    _input_type = AudioStim


class AudioTrimmingFilter(AudioFilter):

    ''' Temporally trims the contents of the audio stimulus using the provided
    start and end points.

    Args:
        start (float): New start point for the trimmed video in seconds.
        end (float): New end point for the trimmed video in seconds.
    '''

    _log_attributes = ('start', 'end')

    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end
        super(AudioTrimmingFilter, self).__init__()

    def _filter(self, audio):
        subclip = audio.clip.subclip(self.start, self.end)
        return AudioStim(onset=audio.onset, filename=audio.filename, clip=subclip)
