from featurex.stimuli import audio
from featurex.stimuli import text
from featurex.converters import Converter

class AudioToTextConverter(Converter):

    ''' Base AudioToText Converter class; all subclasses can only be applied to
    audio and convert to text. '''
    target = audio.AudioStim
    _output_type = text.TextStim