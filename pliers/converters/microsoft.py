''' Microsoft Azure API-based Converter classes. '''

from .image import ImageToTextConverter
from pliers.stimuli.text import TextStim
from pliers.transformers import MicrosoftVisionAPITransformer


class MicrosoftAPITextConverter(MicrosoftVisionAPITransformer, ImageToTextConverter):

    ''' Detects text within images using the Microsoft Vision API. '''

    api_method = 'ocr'
    _log_attributes = ('api_version', 'language')

    def __init__(self, language='en', **kwargs):
        self.language = language
        super(MicrosoftAPITextConverter, self).__init__(**kwargs)

    def _convert(self, stim):
        params = {
            'language': self.language,
            'detectOrientation': False
        }
        response = self._query_api(stim, params)

        lines = []
        for r in response['regions']:
            for l in r['lines']:
                lines.append(' '.join([w['text'] for w in l['words']]))

        text = '\n'.join(lines)
        return TextStim(text=text, onset=stim.onset, duration=stim.duration)
