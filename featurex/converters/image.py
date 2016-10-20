from featurex.stimuli.image import ImageStim
from featurex.stimuli.text import TextStim
from featurex.converters import Converter

class ImageToTextConverter(Converter):

    ''' Base ImageToText Converter class; all subclasses can only be applied to
    image and convert to text. '''
    target = ImageStim
    _output_type = TextStim