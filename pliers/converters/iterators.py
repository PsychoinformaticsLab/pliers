from pliers.stimuli.video import VideoStim, DerivedVideoStim
from pliers.stimuli.image import ImageStim
from pliers.stimuli.text import ComplexTextStim, TextStim
from .base import Converter


class StimCollectionIterator(Converter):

    def _convert(self, stim):
        return stim.__iter__()


class VideoFrameIterator(StimCollectionIterator):

    _input_type = VideoStim
    _output_type = ImageStim


class DerivedVideoFrameIterator(StimCollectionIterator):

    # TODO: use VideoFrameIterator for both VideoStim and DerivedVideoStim,
    # but this may require reworking _input_type to handle disjunction rather
    # than the current conjunction, or making the get_converter() code walk
    # up the hierarchy and use superclass iterators.

    _input_type = DerivedVideoStim
    _output_type = ImageStim


class ComplexTextIterator(StimCollectionIterator):

    _input_type = ComplexTextStim
    _output_type = TextStim
