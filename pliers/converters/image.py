''' Converter classes that operate on ImageStim inputs. '''

from PIL import Image
from .base import Converter
from pliers.stimuli.image import ImageStim
from pliers.stimuli.text import TextStim
from pliers.utils import attempt_to_import, verify_dependencies

pytesseract = attempt_to_import('pytesseract')


class ImageToTextConverter(Converter):

    ''' Base ImageToText Converter class; all subclasses can only be applied to
    image and convert to text. '''
    _input_type = ImageStim
    _output_type = TextStim


class TesseractConverter(ImageToTextConverter):

    ''' Uses the Tesseract library to extract text from images. '''

    VERSION = '1.0'

    def _convert(self, stim):
        verify_dependencies(['pytesseract'])
        text = pytesseract.image_to_string(Image.fromarray(stim.data))
        return TextStim(text=text)
