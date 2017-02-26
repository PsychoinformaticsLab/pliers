''' Converter classes that take StimCollection classes and return their
constituent elements as iterables. '''

from pliers.stimuli.video import VideoStim, DerivedVideoStim
from pliers.stimuli.image import ImageStim
from pliers.stimuli.text import ComplexTextStim, TextStim
from .base import Converter


class StimCollectionIterator(Converter):

    ''' Base class for all StimCollectionIterators. '''

    def _convert(self, stim):
        return stim.__iter__()


class VideoFrameIterator(StimCollectionIterator):

    ''' Iterates frames in a VideoStim as ImageStims. '''
    _input_type = VideoStim
    _output_type = ImageStim


class DerivedVideoFrameIterator(StimCollectionIterator):

    ''' Iterates frames in a DerivedVideoStim as ImageStims. '''

    # TODO: use VideoFrameIterator for both VideoStim and DerivedVideoStim,
    # but this may require reworking _input_type to handle disjunction rather
    # than the current conjunction, or making the get_converter() code walk
    # up the hierarchy and use superclass iterators.

    _input_type = DerivedVideoStim
    _output_type = ImageStim


class ComplexTextIterator(StimCollectionIterator):

    ''' Iterates elements in a ComplexTextStim as TextStims. '''

    _input_type = ComplexTextStim
    _output_type = TextStim
