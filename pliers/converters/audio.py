''' Converters that operate on AudioStim inputs. '''

from pliers.stimuli.audio import AudioStim
from pliers.stimuli.text import TextStim
from .base import Converter


class AudioToTextConverter(Converter):

    ''' Base AudioToText Converter class; all subclasses can only be applied to
    audio and convert to text. '''
    _input_type = AudioStim
    _output_type = TextStim
