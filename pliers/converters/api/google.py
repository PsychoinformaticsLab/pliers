''' Google-based Converter classes. '''

import base64
import os
import tempfile

from pliers.converters.audio import AudioToTextConverter
from pliers.converters.image import ImageToTextConverter
from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.transformers import (GoogleVisionAPITransformer,
                                 GoogleAPITransformer)


class GoogleSpeechAPIConverter(GoogleAPITransformer, AudioToTextConverter):

    ''' Uses the Google Speech API to do speech-to-text transcription.

    Args:
        language_code (str): The language of the supplied AudioStim.
        profanity_filter (bool): If set to True, will ask Google to try and
            filter out profanity from the resulting Text.
        speech_contexts (list): A list of favored phrases or words
+            to assist the API.
        discovery_file (str): path to discovery file containing Google
            application credentials.
        api_version (str): API version to use.
        max_results (int): Max number of results per page.
        num_retries (int): Number of times to retry query on failure.
        rate_limit (int): The minimum number of seconds required between
                transform calls on this Transformer.
    '''

    api_name = 'speech'
    resource = 'speech'
    _log_attributes = ('discovery_file', 'language_code', 'profanity_filter',
                       'speech_contexts')

    def __init__(self, language_code='en-US', profanity_filter=False,
                 speech_contexts=None, discovery_file=None, api_version='v1',
                 max_results=100, num_retries=3, rate_limit=None):
        self.language_code = language_code
        self.profanity_filter = profanity_filter
        self.speech_contexts = speech_contexts
        super(GoogleSpeechAPIConverter,
              self).__init__(discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)

    def _query_api(self, request):
        request_obj = self.service.speech().recognize(body=request)
        return request_obj.execute(num_retries=self.num_retries)

    def _build_request(self, stim):
        tmp = tempfile.mktemp() + '.flac'
        stim.clip.write_audiofile(tmp, fps=stim.sampling_rate, codec='flac',
                                  ffmpeg_params=['-ac', '1'])

        with open(tmp, 'rb') as f:
            data = f.read()
        os.remove(tmp)

        if self.speech_contexts:
            speech_contexts = [{'phrases': self.speech_contexts}]
        else:
            speech_contexts = []
        request = {
            'audio': {
                'content': base64.b64encode(data).decode()
            },
            'config': {
                'encoding': 'FLAC',
                'sampleRateHertz': stim.sampling_rate,
                'languageCode': self.language_code,
                'maxAlternatives': 1,
                'profanityFilter': self.profanity_filter,
                'speechContexts': speech_contexts,
                'enableWordTimeOffsets': True
            }
        }

        return request

    def _convert(self, stim):
        request = self._build_request(stim)
        response = self._query_api(request)

        if 'error' in response:
            raise Exception(response['error']['message'])

        words = []
        if 'results' in response:
            for result in response['results']:
                transcription = result['alternatives'][0]
                for w in transcription['words']:
                    onset = float(w['startTime'][:-1])
                    duration = float(w['endTime'][:-1]) - onset
                    words.append(TextStim(text=w['word'],
                                          onset=onset,
                                          duration=duration))

        return ComplexTextStim(elements=words)


class GoogleVisionAPITextConverter(GoogleVisionAPITransformer,
                                   ImageToTextConverter):

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
    VERSION = '1.0'
    _log_attributes = ('discovery_file', 'handle_annotations', 'api_version')

    def __init__(self, handle_annotations='first', discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        self.handle_annotations = handle_annotations
        super(GoogleVisionAPITextConverter,
              self).__init__(discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)

    def _convert(self, stims):
        request = self._build_request(stims)
        responses = self._query_api(request)
        texts = []

        for response in responses:
            if response and self.response_object in response:
                annotations = response[self.response_object]
                # Combine the annotations
                if self.handle_annotations == 'first':
                    text = annotations[0]['description']
                    texts.append(TextStim(text=text))
                elif self.handle_annotations == 'concatenate':
                    text = ''
                    for annotation in annotations:
                        text = ' '.join([text, annotation['description']])
                    texts.append(TextStim(text=text))
                elif self.handle_annotations == 'list':
                    for annotation in annotations:
                        texts.append(TextStim(text=annotation['description']))
            elif 'error' in response:
                raise Exception(response['error']['message'])
            else:
                texts.append(TextStim(text=''))

        return texts
