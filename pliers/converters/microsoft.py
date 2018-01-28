''' Microsoft Azure API-based Converter classes. '''

from .image import ImageToTextConverter
from pliers.stimuli.text import TextStim
from pliers.transformers import MicrosoftVisionAPITransformer


class MicrosoftAPITextConverter(MicrosoftVisionAPITransformer, ImageToTextConverter):

    ''' Detects text within images using the Microsoft Vision API. '''

    api_method = 'ocr'

    def _convert(self, stim):
        with stim.get_filename() as filename:
            data = open(filename, 'rb').read()

        params = {
            'language': 'en',
            'detectOrientation': False
        }
        response = self._query_api(data, params)

        text = ''
        for r in response['regions']:
            for l in r['lines']:
                for w in l['words']:
                    if text:
                        text += ' ' + w['text']
                    else:
                        text += w['text']

        return TextStim(text=text, onset=stim.onset, duration=stim.duration)
