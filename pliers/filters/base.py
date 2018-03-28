''' Base Filter class and associated functionality. '''

from abc import ABCMeta, abstractmethod
from six import with_metaclass
from pliers.stimuli import AudioStim, VideoStim
from pliers.transformers import Transformer
from pliers.utils import listify

import logging


class Filter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Filters.'''

    def _transform(self, stim, *args, **kwargs):
        new_stim = self._filter(stim, *args, **kwargs)
        if not isinstance(new_stim, self._input_type) and \
           not isinstance(listify(new_stim)[0], stim.__class__):
            raise ValueError("Filter must return a Stim of the same type as "
                             "its input.")
        return new_stim

    @abstractmethod
    def _filter(self, stim):
        pass


class TemporalTrimmingFilter(Filter):
    ''' Temporally trims the contents of the audio stimulus using the provided
    start and end points.

    Args:
        start (float): New start point for the trimmed video in seconds.
        end (float): New end point for the trimmed video in seconds.
        frames (bool): If True, treat the provided start and end values as
            frame indices, otherwise, treat them as time in terms of seconds.
            If True, start and end must both be integers.
        validation (str): String specifying how OOB errors (end > duration)
            should be handled. Must be one of:
                - 'strict': Raise an exception on an OOB error
                - 'warn': Issue a warning for all OOB errors and use the
                maximum bounds
    '''

    _log_attributes = ('start', 'end', 'frames', 'validation')
    _input_type = ()
    _optional_input_type = (AudioStim, VideoStim)

    def __init__(self, start=0, end=None, frames=False, validation='warn'):
        self.start = start
        self.end = end
        self.frames = frames
        self.validation = validation
        super(TemporalTrimmingFilter, self).__init__()

    def _filter(self, stim):
        rate = 'fps' if isinstance(stim, VideoStim) else 'sampling_rate'
        start = self.start / getattr(stim, rate) if self.frames else self.start
        end = self.end / getattr(stim, rate) if self.frames else self.end
        if end and end > stim.duration:
            if self.validation == 'warn':
                logging.warn("Attempted to trim beyond the duration of the"
                             "clip, instead trimming to the end of the clip")
                end = stim.duration
            else:
                raise ValueError("Invalid end argument passed: Attempted to"
                                 "trim beyond the duration of the clip")
        subclip = stim.clip.subclip(start, end)
        return type(stim)(onset=stim.onset, filename=stim.filename, clip=subclip)
