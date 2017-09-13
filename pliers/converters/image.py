''' Converter classes that operate on ImageStim inputs. '''

from PIL import Image
from .base import Converter
from pliers.stimuli.image import ImageStim
from pliers.stimuli.text import TextStim
from pliers.support.decorators import requires_optional_dependency
from pliers.utils import attempt_to_import

pytesseract = attempt_to_import('pytesseract')


class ImageToTextConverter(Converter):

    ''' Base ImageToText Converter class; all subclasses can only be applied to
    image and convert to text. '''
    _input_type = ImageStim
    _output_type = TextStim


class TesseractConverter(ImageToTextConverter):

    ''' Uses the Tesseract library to extract text from images. '''

    @requires_optional_dependency('pytesseract')
    def _convert(self, stim):
        text = pytesseract.image_to_string(Image.fromarray(stim.data))
        return TextStim(text=text, onset=stim.onset, duration=stim.duration)
