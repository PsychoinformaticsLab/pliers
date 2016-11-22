from featurex.stimuli.image import ImageStim
from featurex.stimuli.text import TextStim
from featurex.converters import Converter

from PIL import Image

class ImageToTextConverter(Converter):

    ''' Base ImageToText Converter class; all subclasses can only be applied to
    image and convert to text. '''
    target = ImageStim
    _output_type = TextStim


class TesseractConverter(ImageToTextConverter):
    ''' Uses the Tesseract library to extract text from images '''

    def _convert(self, image):
        import pytesseract
        text = pytesseract.image_to_string(Image.fromarray(image.data))
        return TextStim(text=text)