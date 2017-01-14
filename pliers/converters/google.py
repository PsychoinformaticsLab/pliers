from .image import ImageToTextConverter
from pliers.stimuli.text import TextStim
from pliers.google import GoogleVisionAPITransformer


class GoogleVisionAPITextConverter(GoogleVisionAPITransformer, ImageToTextConverter):

    request_type = 'TEXT_DETECTION'
    response_object = 'textAnnotations'

    def __init__(self, handle_annotations='first', **kwargs):
        super(GoogleVisionAPITextConverter, self).__init__(**kwargs)
        self.handle_annotations = handle_annotations


    def _convert(self, stim):
        request =  self._build_request([stim])
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
