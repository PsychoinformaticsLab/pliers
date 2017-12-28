''' Google-based Converter classes. '''

import base64
import os
import tempfile

from .audio import AudioToTextConverter
from .image import ImageToTextConverter
from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.google import GoogleVisionAPITransformer, GoogleAPITransformer


class GoogleSpeechAPIConverter(GoogleAPITransformer, AudioToTextConverter):

    ''' Uses the Google Speech API to do speech-to-text transcription

    Args:
        language_code (str): The language of the supplied AudioStim.
        profanity_filter (bool): If set to True, will ask Google to try and
            filter out profanity from the resulting Text.
        speech_contexts (list): A list of a list of favored phrases or words
            to assist the API. The inner list is a sequence of word tokens,
            each outer element is a potential context.
    '''

    api_name = 'speech'
    resource = 'speech'
    _log_attributes = ('language_code', 'profanity_filter', 'speech_contexts',
                       'handle_annotations')

    def __init__(self, language_code='en-US', profanity_filter=False,
                 speech_contexts=None, *args, **kwargs):
        self.language_code = language_code
        self.profanity_filter = profanity_filter
        self.speech_contexts = speech_contexts
        super(GoogleSpeechAPIConverter, self).__init__(*args, **kwargs)

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

        content = base64.b64encode(data).decode()
        if self.speech_contexts:
            speech_contexts = [{'phrases': c} for c in self.speech_contexts]
        else:
            speech_contexts = []
        request = {
            'audio': {
                'content': content
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

        offset = 0.0 if stim.onset is None else stim.onset
        if 'results' in response:
            for result in response['results']:
                transcription = result['alternatives'][0]
                words = []
                for w in transcription['words']:
                    onset = float(w['startTime'][:-1])
                    duration = float(w['endTime'][:-1]) - onset
                    words.append(TextStim(text=w['word'],
                                          onset=offset + onset,
                                          duration=duration))

        return ComplexTextStim(elements=words, onset=stim.onset)


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
    VERSION = '1.0'

    def __init__(self, handle_annotations='first', *args, **kwargs):
        super(GoogleVisionAPITextConverter, self).__init__(*args, **kwargs)
        self.handle_annotations = handle_annotations

    def _convert(self, stims):
        request = self._build_request(stims)
        responses = self._query_api(request)
        texts = []

        for i, response in enumerate(responses):
            stim = stims[i]
            if response and self.response_object in response:
                annotations = response[self.response_object]
                # Combine the annotations
                if self.handle_annotations == 'first':
                    text = annotations[0]['description']
                    texts.append(TextStim(text=text, onset=stim.onset,
                                 duration=stim.duration))
                elif self.handle_annotations == 'concatenate':
                    text = ''
                    for annotation in annotations:
                        text = ' '.join([text, annotation['description']])
                    texts.append(TextStim(text=text, onset=stim.onset,
                                 duration=stim.duration))
                elif self.handle_annotations == 'list':
                    for annotation in annotations:
                        texts.append(TextStim(text=annotation['description'],
                                              onset=stim.onset,
                                              duration=stim.duration))
            elif 'error' in response:
                raise Exception(response['error']['message'])
            else:
                texts.append(TextStim(text='', onset=stim.onset, duration=stim.duration))

        return texts
