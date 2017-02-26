''' Google-based Converter classes. '''

from .image import ImageToTextConverter
from pliers.stimuli.text import TextStim
from pliers.google import GoogleVisionAPITransformer


class GoogleVisionAPITextConverter(GoogleVisionAPITransformer, ImageToTextConverter):

    ''' Detects text within images using the Google Cloud Vision API.
    Args:
        handle_annotations (str): How to handle cases where there are multiple
            detected text labels. Valid values are 'first' (only return the
            first response as a TextStim), 'concatenate' (concatenate all
            responses into a single TextStim), or 'list' (return a list of
            TextStims).
        args, kwargs: Optional positional and keyword arguments to pass to
            the superclass init.
    '''

    request_type = 'TEXT_DETECTION'
    response_object = 'textAnnotations'

    def __init__(self, handle_annotations='first', *args, **kwargs):
        super(GoogleVisionAPITextConverter, self).__init__(*args, **kwargs)
        self.handle_annotations = handle_annotations

    def _convert(self, stim):
        request = self._build_request([stim])
        responses = self._query_api(request)
        response = responses[0]

        if response:
            annotations = response[self.response_object]
            # Combine the annotations
            if self.handle_annotations == 'first':
                text = annotations[0]['description']
                return TextStim(text=text, onset=stim.onset,
                                duration=stim.duration)
            elif self.handle_annotations == 'concatenate':
                text = ''
                for annotation in annotations:
                    text += annotation['description']
                return TextStim(text=text, onset=stim.onset,
                                duration=stim.duration)
            elif self.handle_annotations == 'list':
                texts = []
                for annotation in annotations:
                    texts.append(TextStim(text=annotation['description'],
                                          onset=stim.onset,
                                          duration=stim.duration))
                return texts

        else:
            return TextStim(text='', onset=stim.onset, duration=stim.duration)
