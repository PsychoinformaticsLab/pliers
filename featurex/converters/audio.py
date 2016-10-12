from featurex.stimuli.audio import AudioStim
from featurex.stimuli.text import TextStim
from featurex.converters import Converter

class AudioToTextConverter(Converter):

    ''' Base AudioToText Converter class; all subclasses can only be applied to
    audio and convert to text. '''
    target = AudioStim
    _output_type = TextStim